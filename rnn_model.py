import torch
import torch.nn as nn
import numpy as np

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

from sample import NegativeSampler

from utils.utils import repackage_hidden

from eucl_distance.distance import eucl_distance, dot_distance
from rnncells import LinearRNNCell, ELURNNCell, DExpRNNCell
from threshold import hard_threshold, soft_threshold1, soft_threshold2, DynamicThreshold

class TLModel(nn.Module):
    """Container module with an encoder and a recurrent module."""

    def __init__(self, ntoken, rnn_config, reg_config, sample_config, threshold_config, bucket_config):

        initrange = 0.1
        super(TLModel, self).__init__()
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

        # bucket config
        self.buckets = [0] + bucket_config.buckets + [1000000]
        self.tombstones = torch.LongTensor([self.ntoken + i for i in range(len(self.buckets)-1)]).cuda()
        self.nbuckets = len(self.buckets)-1

        # initialize cell
        if self.cell_type == 'rnn': self.rnn = torch.nn.RNN(self.ninp, self.nhid, 1, dropout=0)
        if self.cell_type == 'linear_rnn': self.rnn = LinearRNNCell(self.ninp, self.nhid, dropout=0)
        if self.cell_type == 'relu_rnn': self.rnn = torch.nn.RNN(self.ninp, self.nhid, 1, dropout=0, nonlinearity='relu')
        if self.cell_type == 'elu_rnn': self.rnn = ELURNNCell(self.ninp, self.nhid, dropout=0)
        if self.cell_type == 'dexp_rnn': self.rnn = DExpRNNCell(self.ninp, self.nhid, dropout=0)
        if self.cell_type == 'gru': self.rnn = torch.nn.GRU(self.ninp, self.nhid, 1, dropout=0)

        # set distance function
        self.dist_fn = eucl_distance

        # initialize encoder
        self.encoder = nn.Embedding(self.ntoken + len(self.buckets) - 1, self.ninp)
        self.encoder = self.init_weights(self.encoder, initrange=initrange, path=rnn_config.word_embeddings_path)
        self.fixed_word_embeddings = (not rnn_config.word_embeddings_path is None)
        
        # initialize bias
        self.decoder = nn.Linear(self.nhid, ntoken + len(self.buckets) - 1)
        self.decoder = self.init_weights(self.decoder, initrange=initrange)
        self.bias = self.decoder.bias

        # initialize dropouts and apply weight drop
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(self.dropouti)
        self.hdrop = nn.Dropout(self.dropouth)
        self.drop = nn.Dropout(self.dropout)
        self.rnn = WeightDrop(self.rnn, ['weight_hh_l0'], dropout=reg_config.wdrop)

        # initialize sampler
        self.sampler = NegativeSampler(self.nsamples, torch.ones(self.ntoken) if sample_config.frequencies is None else sample_config.frequencies)

        # initialize threshold 
        if self.threshold_func == 'hard': self.treshold = hard_threshold
        if self.threshold_func == 'soft1': self.threshold = soft_threshold1
        if self.threshold_func == 'soft2': self.threshold = soft_threshold2
        if self.threshold_func == 'dynamic': self.threshold = DynamicThreshold(self.nhid, self.threshold_nhid, self.threshold_nlayers, self.threshold_temp)
        if self.threshold_func == 'none': self.threshold = None
        self.inf = 1e8

        # more biases
        self.lin_b_h = nn.Linear(self.nhid, 1)
        self.b_h = self.lin_b_h.bias
        self.lin_b_w = nn.Linear(self.nhid, self.ntoken + len(self.buckets) - 1)
        self.b_w = self.lin_b_w.bias
        
      
    def init_weights(self, module, initrange=0.1, path=None):
        if path is None:
            module.weight.data.uniform_(-initrange, initrange)
        else:
            weights = np.loadtxt(path)
            module.weight.data = torch.FloatTensor(weights).cuda()
        return module

    def _apply_threshold(self, d, h, b_w=None):
        '''
            d: pairwise distances between h and h_+
            h: initial hidden states h
        '''
        #return d
        #alpha = 0.1
        #return (torch.exp(alpha * d) - 1)/alpha
        #return torch.clamp((torch.exp(alpha*(d + b_w + self.b_h))-1)/alpha, max=self.inf)
        # return d if no thresholding necessary
        if self.threshold_mode == 'none':
            #print(self.threshold_mode)
            return d
        if self.threshold_mode == 'train' and not self.training:
            return d
        if self.threshold_mode == 'eval' and self.training:
            return d

        # two cases: either dynamic or fixed radius
        if self.threshold_func == 'dynamic':
            d, r = self.threshold(d + b_w + self.b_h, h, self.inf)
            print(d.min(), d.max(), r.min(), r.max())
        else:
            d = self.threshold(d + b_w + self.b_h, self.threshold_max_r, self.inf)
        return d

    def _apply_temperature(self, d):
        return self.beta * d

    def _apply_bias(self, d, b):
        return d + b

    def _forward(self, words_times_W, hiddens_times_U, hidden=None):

        tanh, sigmoid = nn.Tanh(), nn.Sigmoid()

        if self.cell_type == 'rnn': output = torch.nn.functional.tanh(words_times_W + hiddens_times_U)
        if self.cell_type == 'linear_rnn': output = words_times_W + hiddens_times_U
        if self.cell_type == 'relu_rnn': output = torch.nn.functional.relu(words_times_W + hiddens_times_U)
        if self.cell_type == 'elu_rnn': output = torch.nn.functional.elu(words_times_W + hiddens_times_U)
        if self.cell_type == 'dexp_rnn': output = self.rnn.module.dilated_exp(words_times_W + hiddens_times_U)
        if self.cell_type == 'gru':
            _ir = torch.nn.functional.sigmoid(words_times_W[:,:self.nhid] + hiddens_times_U[:,:self.nhid])
            _iz = torch.nn.functional.sigmoid(words_times_W[:,self.nhid:2*self.nhid] + hiddens_times_U[:,self.nhid:2*self.nhid])
            _in = torch.nn.functional.tanh(words_times_W[:,2*self.nhid:] + _ir * hiddens_times_U[:,2*self.nhid:])
            output = (1 - _iz) * _in + _iz * hidden

        return output

    def _data2bucket(self, data, argsort):

        data2 = data.clone(); data2[data2 >= 10000] = 9999                      # ugly fix for padding tokens!!
        data2 = torch.cat([(a == d).nonzero() for a, d in zip(argsort,data2)])  # index of data in sorted array
        mask = None
        for idx in range(0, self.nbuckets):
            partial_mask = data2 >= self.buckets[idx+1]
            mask = mask + partial_mask.long() if mask is not None else partial_mask.long()
        return mask
        #buckets = data.clone() // 100
        #buckets[buckets >= len(self.buckets)-1] = 0
        #return buckets

    def _get_tombstones(self, argsort):

        nbuckets = self.nbuckets
        ndynamic_buckets = 12   # atm the first few

        seq_len_times_bsz = argsort.size(0)

        tombstones_emb, tombstones_bias = [], []
        # tombstones for ndynamic buckets are the mean of the word embeddings in the bucket
        for i in range(ndynamic_buckets):
            idxs = argsort[:, self.buckets[i]:self.buckets[i+1]] if i < self.nbuckets-1 else argsort[:, self.buckets[i]:]   # (seq_len x bsz) x bucket_size
            emb = embedded_dropout(self.encoder, idxs, dropout=self.dropoute if self.training else 0)                       # (seq_len x bsz) x bucket_size x emb
            tombstones_emb.append(emb.mean(1).view(-1, 1, self.ninp))                                                       # (seq_len x bsz) x 1 x emb
            tombstones_bias.append(self.bias[idxs].mean(1).view(-1, 1))                                             # (seq_len x bsz) x 1

        for i in range(ndynamic_buckets, nbuckets):
            idx = torch.LongTensor([-(nbuckets - ndynamic_buckets - i)]).cuda().repeat(seq_len_times_bsz)                   # (seq_len x bsz)
            emb = embedded_dropout(self.encoder, idx, dropout=self.dropoute if self.training else 0)
            tombstones_emb.append(emb.view(-1, 1, self.ninp))                                                               # (seq_len x bsz) x 1 x emb 
            tombstones_bias.append(self.bias[idx].view(-1, 1))                                                             # (seq_len x bsz) x 1

        tombstones_emb = torch.cat(tombstones_emb, 1)   # (seq_len x bsz) x ntombstones x emb
        tombstones_bias = torch.cat(tombstones_bias, 1) # (seq_len x bsz) x ntombstones

        return tombstones_emb, tombstones_bias

    def _sample_from_bucket(self, data, argsort, eval_whole_bucket=False):

        data = data.view(-1)
        if eval_whole_bucket:
            # NEED SAME SIZE BUCKETS
            bucket_size = self.buckets[1] - self.buckets[0]
            nbuckets = len(self.buckets) - 1

            #print(data, nbuckets, bucket_size, [i for i in range(self.ntoken-1)])

            samples_per_bucket = torch.LongTensor([i for i in range(self.ntoken-1)]).cuda().view(nbuckets, -1).cuda()
            idxs = self._data2bucket(data)

            samples = samples_per_bucket[idxs]
            return samples # size (seq_len x bsz) x bucketsize

        else:
            all_words = torch.LongTensor([i for i in range(self.ntoken)]).cuda()
            samples_per_bucket = torch.zeros(self.nbuckets, self.nsamples).long().cuda()
            for i in range(self.nbuckets):
                bucket_size = self.buckets[i+1] - self.buckets[i] if i < self.nbuckets-1 else self.ntoken - self.buckets[i]
                weights = torch.ones(bucket_size).cuda()
                sampler = torch.utils.data.WeightedRandomSampler(weights, self.nsamples)
                samples_per_bucket[i] = torch.LongTensor(list(sampler)).cuda() + self.buckets[i]

            idxs = self._data2bucket(data, argsort)
            samples = samples_per_bucket[idxs.view(-1)]

            # "translate samples"
            for i in range(argsort.size(0)):
                samples[i] = argsort[i, samples[i]]
            return samples

    def _logsoftmax_over_tombstones(self, bucket_idxs, raw_output, ts_emb, ts_bias):

        seq_len_times_bsz = raw_output.size(0)
        ntombstones = self.nbuckets
         
        # only one layer for the moment
        weights_ih, bias_ih = self.rnn.module.weight_ih_l0, self.rnn.module.bias_ih_l0  
        weights_hh, bias_hh = self.rnn.module.weight_hh_l0, self.rnn.module.bias_hh_l0

        # reshape samples for indexing and precompute the inputs to nonlinearity
        ts_times_W = torch.nn.functional.linear(ts_emb, weights_ih, bias_ih) # (seq_len x _bsz) x ntombstones x nhid
        hiddens_times_U = torch.nn.functional.linear(raw_output, weights_hh, bias_hh) # (seq_len x bsz) x nhid
        print(ts_times_W.size(), hiddens_times_U.size())

        # iterate over samples to update loss
        x = torch.zeros(ntombstones, seq_len_times_bsz).cuda()
        for i in range(ntombstones):

            # compute output of negative samples
            output = self._forward(ts_times_W[:,i,:], hiddens_times_U, raw_output)
            output = self.lockdrop(output.view(1, output.size(0), -1), self.dropout)
            output = output[0]

            # compute loss term
            d_neg = self.dist_fn(raw_output, output)

            if not self.threshold is None:
                pass#d_neg = self._apply_threshold(d_neg, raw_output, self.b_w[ts[i]])
            d_neg = self._apply_temperature(d_neg)
            d_neg = self._apply_bias(d_neg, ts_bias[:,i])
        
            x[i] = -d_neg
        
        softmaxed = torch.nn.functional.log_softmax(x, dim=0)
        print(bucket_idxs.size(), softmaxed.size())
        softmaxed = softmaxed.gather(0, bucket_idxs.view(1,-1))

        return softmaxed
        
    def _logsoftmax_over_neg_samples(self, d_pos, raw_output, samples, samples_emb):

        seq_len, bsz = d_pos.size()
        nsamples = samples.size(1)

        # positive sample distance
        x = torch.zeros(1 + nsamples, seq_len*bsz).cuda()
        x[0] = -d_pos.view(seq_len * bsz)
        
        # only one layer for the moment
        weights_ih, bias_ih = self.rnn.module.weight_ih_l0, self.rnn.module.bias_ih_l0  
        weights_hh, bias_hh = self.rnn.module.weight_hh_l0, self.rnn.module.bias_hh_l0

        # reshape samples for indexing and precompute the inputs to nonlinearity
        samples_times_W = torch.nn.functional.linear(samples_emb, weights_ih, bias_ih)
        hiddens_times_U = torch.nn.functional.linear(raw_output, weights_hh, bias_hh)

        print(samples_times_W.size(), hiddens_times_U.size())
        
        # iterate over samples to update loss
        for i in range(nsamples):

            # compute output of negative samples
            output = self._forward(samples_times_W[:,i,:], hiddens_times_U, raw_output)
            output = self.lockdrop(output.view(1, output.size(0), -1), self.dropout)
            output = output[0]

            # compute loss term
            d_neg = self.dist_fn(raw_output, output)

            if not self.threshold is None:
                pass#d_neg = self._apply_threshold(d_neg, raw_output, self.b_w[samples[:,i]])
            d_neg = self._apply_temperature(d_neg)
            d_neg = self._apply_bias(d_neg, self.bias[samples[:,i]])
        
            x[i+1] = -d_neg
        
        softmaxed = torch.nn.functional.log_softmax(x, dim=0)[0]
        return softmaxed
    
    def forward(self, data, binary, hidden, argsort):

        # argsort has size (seq_len x bsz) x ntokens

        # get batch size and sequence length
        seq_len, bsz = data.size()

        # pass sequence through rnn
        emb = embedded_dropout(self.encoder, data, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)
        if self.fixed_word_embeddings:
            emb = emb.detach()

        raw_output, new_hidden = self.rnn(emb, hidden)          # apply single layer rnn
        raw_output = self.lockdrop(raw_output, self.dropout)    # seq_len x bsz x nhid
        raw_output = raw_output.view(seq_len, bsz, -1)          # reshape for concat
        raw_output = torch.cat((hidden, raw_output), 0)         # concatenate initial hidden state

        # find distance between positive samples 
        d_pos = (raw_output[1:] - raw_output[:-1]).norm(dim=2).pow(2)
        if not self.threshold is None:
            d_pos = self._apply_threshold(d_pos, raw_output[:-1], self.b_w[data])
        d_pos = self._apply_temperature(d_pos)
        d_pos = self._apply_bias(d_pos, self.bias[data])

        
        # hiddens used for negative sampling are all except last
        raw_output = raw_output[:-1].view(seq_len*bsz, -1) 
        
        # softmax over tombstones
        tombstones_emb, tombstones_bias = self._get_tombstones(argsort)
        tombstones_emb = self.lockdrop(tombstones_emb, self.dropouti)
        ts_softmaxed = self._logsoftmax_over_tombstones(self._data2bucket(data.view(-1), argsort), raw_output, tombstones_emb, tombstones_bias) 
        
        # softmax over negative samples
        samples = self._sample_from_bucket(data, argsort)
        samples_emb = embedded_dropout(self.encoder, samples, dropout=self.dropoute if self.training else 0)
        samples_emb = self.lockdrop(samples_emb, self.dropouti)
        
        ns_softmaxed = self._logsoftmax_over_neg_samples(d_pos, raw_output, samples, samples_emb)
        
        # overall softmax is sum of the two
        softmaxed = ts_softmaxed + ns_softmaxed
        softmax_mapped = -softmaxed.view(seq_len, bsz) * binary
        softmax_mapped = softmax_mapped / binary.sum()

        loss = softmax_mapped[softmax_mapped > 0].sum() if softmax_mapped.sum() > 1e-12 else softmax_mapped.mean()
        return loss


    def evaluate(self, data, hidden, argsort, eos_tokens=None, dump_hiddens=False):

        # get weights and compute WX for all words
        weights_ih, bias_ih = self.rnn.module.weight_ih_l0, self.rnn.module.bias_ih_l0  # only one layer for the moment
        weights_hh, bias_hh = self.rnn.module.weight_hh_l0, self.rnn.module.bias_hh_l0

        all_words = torch.LongTensor([i for i in range(self.ntoken)]).cuda()
        all_words = embedded_dropout(self.encoder, all_words, dropout=self.dropoute if self.training else 0)

        all_words_times_W = torch.nn.functional.linear(all_words, weights_ih, bias_ih)
        
        ts_emb, ts_bias = self._get_tombstones(argsort)                         # seq_len x buckets x emb, seq_len
        ts_times_W = torch.nn.functional.linear(ts_emb, weights_ih, bias_ih)    # seq_len x buckets x nhid

        # iterate over data set and compute loss
        total_loss = 0
        i = 0

        entropy, hiddens, all_hiddens = [], [], []
        print(argsort)
        data2 = torch.cat([(a == d).nonzero() for a, d in zip(argsort,data)])
        print(data, data2)
        while i < data.size(0):

            hidden_times_U = torch.nn.functional.linear(hidden[0].repeat(self.ntoken, 1), weights_hh, bias_hh)
            
            # first tombstone probs
            ts_output = self._forward(ts_times_W[i], hidden_times_U[:self.nbuckets], hidden[0].repeat(self.nbuckets, 1))
            distance = self.dist_fn(hidden[0], ts_output)
            if not self.threshold is None:
                pass#distance = self._apply_threshold(distance, hidden[0], self.b_w[self.ntoken:])
            distance = self._apply_temperature(distance)
            distance = self._apply_bias(distance, self.bias[self.ntoken:])

            #print(i, data.size(), data[i] // 10)
            bucket = self._data2bucket(data[i], argsort)
            bucket_size = self.buckets[bucket+1] - self.buckets[bucket] if bucket < self.nbuckets-1 else self.ntoken - self.buckets[bucket]
            softmaxed = torch.nn.functional.log_softmax(-distance, dim=0)
            #print(left_idx, softmaxed)
            raw_loss = -softmaxed[bucket].item()   # TODOOOO

            all_words_times_W_i = all_words_times_W[argsort[i]]
            #print(all_words_times_W_i.size())
            output = self._forward(all_words_times_W[self.buckets[bucket]:self.buckets[bucket]+bucket_size], hidden_times_U[:bucket_size], hidden[0].repeat(bucket_size, 1))

            if dump_hiddens: pass#hiddens.append(output[data[i]].data.cpu().numpy())

            distance = self.dist_fn(hidden[0], output)     
            if not self.threshold is None:
                pass#distance = self._apply_threshold(distance, hidden[0], self.b_w[self.buckets[bucket]:self.buckets[bucket]+bucket_size])
            distance = self._apply_temperature(distance)
            distance = self._apply_bias(distance, self.bias[self.buckets[bucket]:self.buckets[bucket]+bucket_size])
       
            softmaxed = torch.nn.functional.log_softmax(-distance, dim=0) 
            print(bucket, self.buckets[bucket], bucket_size, data2[i])
            raw_loss = raw_loss - softmaxed[data2[i] - self.buckets[bucket]].item()

            total_loss += raw_loss / data.size(0)
            entropy.append(raw_loss)

            if not eos_tokens is None and data[i].data.cpu().numpy()[0] in eos_tokens:
                hidden = self.init_hidden(1)
                if dump_hiddens:
                    all_hiddens.append(hiddens)
                    hiddens = []
            else:
                hidden = output[data2[i] - self.buckets[bucket]].view(1, 1, -1)
            hidden = repackage_hidden(hidden)

            i = i + 1

        #all_hiddens = all_hiddens if not eos_tokens is None else hiddens
        
        return total_loss, hidden, np.array(entropy)


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return weight.new(1, bsz, self.nhid).zero_()
