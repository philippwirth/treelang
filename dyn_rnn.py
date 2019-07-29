import torch
import torch.nn as nn
import numpy as np

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

from sample import NegativeSampler

from utils.utils import repackage_hidden

from eucl_distance.distance import eucl_distance
#from rnncells import LinearRNNCell, ELURNNCell, DExpRNNCell, DynamicRNNCell
from dynrnncell import DynamicRNNCell
from threshold import hard_threshold, soft_threshold1, soft_threshold2, DynamicThreshold

class RNNModel(nn.Module):
    """Container module with an encoder and a recurrent module."""

    def __init__(self, ntoken, rnn_config, reg_config, sample_config, threshold_config):

        initrange = 0.1
        super(RNNModel, self).__init__()
        self.ntoken = ntoken

        # rnn configs
        self.cell_type = rnn_config.cell_type   # cell_type in [rnn, linear_rnn, relu_rnn, elu_rnn, dexp_rnn, gru]
        self.ninp = rnn_config.emsize
        self.nhid = rnn_config.nhid
        self.beta = rnn_config.temp

        # regularization configs
        self.dropout = reg_config.dropout
        self.dropouth = reg_config.dropouth
        self.dropouti = reg_config.dropouti
        self.dropoute = reg_config.dropoute
        self.wdrop = reg_config.wdrop

        # sample config
        self.nsamples = sample_config.nsamples
        self.frequencies = sample_config.frequencies

        # threshold config
        self.threshold_func = threshold_config.func
        self.threshold_mode = threshold_config.mode
        self.threshold_temp = threshold_config.temp
        self.threshold_max_r = threshold_config.max_radius
        self.threshold_min_r = threshold_config.min_radius
        self.threshold_decr = threshold_config.decrease_radius
        self.threshold_nlayers = threshold_config.nlayers
        self.threshold_nhid = threshold_config.nhid

        # initialize cell
        self.rnn = DynamicRNNCell(self.ninp, self.nhid, dropout=0)

        # set distance function
        self.dist_fn = eucl_distance

        # initialize encoder
        self.encoder = nn.Embedding(self.ntoken, self.ninp)
        self.encoder = self.init_weights(self.encoder, initrange=initrange, path=rnn_config.word_embeddings_path)
        self.fixed_word_embeddings = (not rnn_config.word_embeddings_path is None)

        # initialize bias
        self.decoder = nn.Linear(self.nhid, ntoken)
        self.decoder = self.init_weights(self.decoder, initrange=initrange)
        self.bias = self.decoder.bias

        # initialize dropouts and apply weight drop
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(self.dropouti)
        self.hdrop = nn.Dropout(self.dropouth)
        self.drop = nn.Dropout(self.dropout)
        self.rnn = WeightDrop(self.rnn, ['weight_hh_l0'], dropout=reg_config.wdrop)

        # initialize sampler
        self.sampler = NegativeSampler(self.nsamples, torch.ones(self.ntoken) if sample_config.frequencies is None else samples_config.frequencies)

        # initialize threshold 
        if self.threshold_func == 'hard': self.treshold = hard_threshold
        if self.threshold_func == 'soft1': self.threshold = soft_threshold1
        if self.threshold_func == 'soft2': self.threshold = soft_threshold2
        if self.threshold_func == 'dynamic': self.threshold = DynamicThreshold(self.nhid, self.threshold_nhid, self.threshold_nlayers, self.threshold_temp)
        if self.threshold_func == 'none': self.threshold = None
        self.inf = 1e8
        
      
    def init_weights(self, module, initrange=0.1, path=None):
        if path is None:
            module.weight.data.uniform_(-initrange, initrange)
        else:
            weights = np.loadtxt(path)
            module.weight.data = torch.FloatTensor(weights).cuda()
        return module

    def _apply_threshold(self, d, h):
        '''
            d: pairwise distances between h and h_+
            h: initial hidden states h
        '''

        # return d if no thresholding necessary
        if self.threshold_mode == 'none':
            return d
        if self.threshold_mode == 'train' and not self.training:
            return d
        if self.threshold_mode == 'eval' and self.training:
            return d

        # two cases: either dynamic or fixed radius
        if self.threshold_func == 'dynamic':
            d, r = self.threshold(d, h, self.inf)
        else:
            d = self.threshold(d, self.threshold_max_r, self.inf)
        return d

    def _apply_temperature(self, d):
        return self.beta * d

    def _apply_bias(self, d, b):
        return d + b

    def _forward(self, words_times_W, hiddens_times_U, hidden=None):

        output = torch.nn.functional.tanh(words_times_W + hiddens_times_U)
        return output

    def forward(self, data, binary, hidden):

        #self.rnn.module.U, self.rnn.module.S, self.rnn.module.V = torch.svd(self.rnn.module.weight_ih_l0)
        # get batch size and sequence length
        seq_len, bsz = data.size()
        
        emb = embedded_dropout(self.encoder, data, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)
        if self.fixed_word_embeddings:
            emb = emb.detach()

        raw_output, new_hidden = self.rnn(emb, hidden)          # apply single layer rnn
        raw_output = self.lockdrop(raw_output, self.dropout)    # seq_len x bsz x nhid
        raw_output = raw_output.view(seq_len, bsz, -1)          # reshape for concat
        raw_output = torch.cat((hidden, raw_output), 0)         # concatenate initial hidden state
        #print(self.rnn.module.weight_ih_l0)

        # initialize loss w/ positive terms
        # compute distances between consecutive hidden states
        d_pos = (raw_output[1:] - raw_output[:-1]).norm(dim=2).pow(2)

        if not self.threshold is None:
            d_pos = self._apply_threshold(d_pos, raw_output[:-1])
        d_pos = self._apply_temperature(d_pos)
        d_pos = self._apply_bias(d_pos, self.bias[data])

        # hiddens used for negative sampling are all except last
        raw_output = raw_output[:-1].view(seq_len*bsz, -1)

        # x stores the positive samples at index 0 and the negative ones a 1:nsamples+1
        x = torch.zeros(1 + self.nsamples, seq_len*bsz).cuda()
        x[0] = -d_pos.view(seq_len * bsz)
        
        # process negative samples
        samples = self.sampler(bsz, seq_len)    # (nsamples x bsz x seq_len)
        samples_emb = embedded_dropout(self.encoder, samples, dropout=self.dropoute if self.training else 0)
        samples_emb = self.lockdrop(samples_emb, self.dropouti).view(self.nsamples, bsz*seq_len, -1)
        if self.fixed_word_embeddings:
            samples_emb = samples_emb.detach()

        # only one layer for the moment
        weights_hh, bias_hh = self.rnn.module.weight_hh_l0, self.rnn.module.bias_hh_l0

        # reshape samples for indexing and precompute the inputs to nonlinearity
        samples = samples.view(self.nsamples, bsz*seq_len)
        hiddens_times_U = torch.nn.functional.linear(raw_output, weights_hh, bias_hh)
        
        # iterate over samples to update loss
        for i in range(self.nsamples):

            # compute output of negative samples
            #print(samples_emb[i].size(), raw_output.size())
            samples_times_W = self.rnn.module._in_times_W(samples_emb[i], raw_output)
            output = self._forward(samples_times_W, hiddens_times_U, raw_output)
            output = self.lockdrop(output.view(1, output.size(0), -1), self.dropout)
            output = output[0]

            # compute loss term
            d_neg = self.dist_fn(raw_output, output)

            if not self.threshold is None:
                d_neg = self._apply_threshold(d_neg, raw_output)
            d_neg = self._apply_temperature(d_neg)
            d_neg = self._apply_bias(d_neg, self.bias[samples[i]])
        
            x[i+1] = -d_neg

        softmaxed = -torch.nn.functional.log_softmax(x, dim=0)[0]
        softmax_mapped = softmaxed.view(seq_len, bsz) * binary
        loss = softmax_mapped.mean()

        #self.rnn.module.weight_ih_l0 = torch.mm(self.rnn.module.U, torch.mm(torch.diag(self.rnn.module.S), self.rnn.module.V.t()))
        return loss


    def evaluate(self, data, eos_tokens=None, dump_hiddens=False):

        # get weights and compute WX for all words
        #weights_ih, bias_ih = self.rnn.module.weight_ih_l0, self.rnn.module.bias_ih_l0  # only one layer for the moment
        #weights_hh, bias_hh = self.rnn.module.weight_hh_l0, self.rnn.module.bias_hh_l0

        all_words = torch.LongTensor([i for i in range(self.ntoken)]).cuda()
        all_words = embedded_dropout(self.encoder, all_words, dropout=self.dropoute if self.training else 0).view(1, self.ntoken, -1)

        # iterate over data set and compute loss
        total_loss, hidden = 0, self.init_hidden(1)
        i = 0

        entropy, hiddens, all_hiddens = [], [], []
        while i < data.size(0):

            #all_words_times_W = self.rnn.module._in_times_W(all_words, hidden)

            #hidden_times_U = torch.nn.functional.linear(hidden[0].repeat(self.ntoken, 1), weights_hh, bias_hh)
            #print(all_words.size(), hidden[0].repeat(1,self.ntoken,1)[0].size())
            output = self.rnn(all_words, hidden[0].repeat(1,self.ntoken,1))[0]#self._forward(all_words_times_W, hidden_times_U, hidden[0].repeat(self.ntoken, 1))

            if dump_hiddens: pass#hiddens.append(output[data[i]].data.cpu().numpy())

            distance = self.dist_fn(hidden[0], output[0])
            #print(output.size(), distance.size(), hidden) 
            if not self.threshold is None:
                distance = self._apply_threshold(distance, hidden[0])
            distance = self._apply_temperature(distance)
            distance = self._apply_bias(distance, self.bias)
        
            softmaxed = torch.nn.functional.log_softmax(-distance, dim=0)
            raw_loss = -softmaxed[data[i]].item()

            total_loss += raw_loss / data.size(0)
            entropy.append(raw_loss)

            if not eos_tokens is None and data[i].data.cpu().numpy()[0] in eos_tokens:
                hidden = self.init_hidden(1)
                hidden = hidden.detach()
                if dump_hiddens:
                    pass#all_hiddens.append(hiddens)
                    hiddens = []
            else:
                hidden = output[0][data[i]].view(1, 1, -1)
            hidden = repackage_hidden(hidden).detach()

            i = i + 1

        all_hiddens = all_hiddens if not eos_tokens is None else hiddens

        if self.threshold_decr > 0:
            self.threshold_max_r = max(self.threshold_min_r, self.threshold_max_r * 0.95)
        
        if dump_hiddens:
            return total_loss, np.array(entropy), all_hiddens
        else:
            return total_loss, np.array(entropy)


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return weight.new(1, bsz, self.nhid).zero_()
