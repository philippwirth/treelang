import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn

from utils import data
from rnn_model import TLModel
from mos.model import RNNModel
#from dyn_rnn import RNNModel

from visualize.dump import dump, dump_hiddens, dump_words
from utils.utils import batchify, batchify_padded, get_batch, repackage_hidden

from collections import namedtuple

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=141,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')

parser.add_argument('--evaluate_every', type=int, default=1)

# dump settings
parser.add_argument('--dump_hiddens', action='store_true')
parser.add_argument('--dump_words', action='store_true')
parser.add_argument('--dump_valloss', type=str, default='valloss')
parser.add_argument('--dump_entropy', type=str, default='entropy_')

args = parser.parse_args()
args.tied = True

from config.rnnconfig import *              # get rnn_config
from config.thresholdconfig import *        # get treshold_config
from config.regularizationconfig import *   # get reg_config
from config.sampleconfig import *           # get sample_config
from config.bucketconfig import *           # get bucket_config

def run(args, rnn_config, reg_config, threshold_config, sample_config, bucket_config):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################

    def model_save(fn):
        with open(fn, 'wb') as f:
            torch.save([tl_model, optimizer], f)

    def model_load(fn):
        global tl_model, criterion, optimizer
        with open(fn, 'rb') as f:
            tl_model, optimizer = torch.load(f)

    import os
    import hashlib
    fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset...')
        corpus = data.Corpus(args.data)
        torch.save(corpus, fn)

    # get token frequencies and eos_tokens
    frequencies, eos_tokens = None, None
    if not sample_config.uniform_freq: sample_config = SampleConfig(sample_config.nsamples, False, corpus.frequencies)
    eos_tokens = corpus.reset_idxs
    print('EOS:' + str(eos_tokens))

    # batchify
    eval_batch_size = 1
    test_batch_size = 1

    ntokens = len(corpus.dictionary) + 1 if args.batch_size > 1 else len(corpus.dictionary)
    train_data, binary_data, seq_lens = batchify_padded(corpus.train, args.batch_size, args, ntokens, eos_tokens)    
    val_data = batchify(corpus.valid, eval_batch_size, args)
    test_data = batchify(corpus.test, test_batch_size, args)
    print('ntokens: ' + str(ntokens))
    print(corpus.frequencies.size())


    ###############################################################################
    # Build the model
    ###############################################################################

    # build treelang model and load mos model
    tl_model = TLModel(ntokens, rnn_config, reg_config, sample_config, threshold_config, bucket_config)
    mos_name = 'mos/TEST-20190731-093115/model.pt'
    with open(mos_name, 'rb') as f:
        mos_model = torch.load(f)

    ###
    if args.resume:
        print('Resuming model ...')
        model_load(args.resume)
        optimizer.param_groups[0]['lr'] = args.lr
        tl_model.dropouti, tl_model.dropouth, tl_model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute

    ###
    if args.cuda:
        tl_model = tl_model.cuda()
        mos_model = mos_model.cuda()

    ###
    params = list(tl_model.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
    print('Args:', args)
    print('Model total parameters:', total_params)

    ###############################################################################
    # Training code
    ###############################################################################

    def evaluate(data_source, epoch, batch_size=1):
        # Turn on evaluation mode which disables dropout.
        tl_model.eval()

        total_loss, i = 0, 0
        h_tl = tl_model.init_hidden(batch_size)
        h_mos = mos_model.init_hidden(batch_size)
        
        while i < data_source.size(0)-1:

            seq_len = args.bptt
            data = get_batch(data_source, i, args, seq_len=seq_len)

            # evaluate mos for probability ranks
            h_mos = repackage_hidden(h_mos)
            log_prob, h_mos = mos_model(data, h_mos)

            # get probability ranks from mos probability
            _, argsort = torch.sort(log_prob, descending=True)
            argsort = argsort.view(-1, 20)##10000) #
            
            # evaluate tl model
            h_tl = repackage_hidden(h_tl)
            loss, h_tl, entropy = tl_model.evaluate(data, h_tl, argsort, eos_tokens)

            total_loss = total_loss + loss*seq_len
            i = i + seq_len

        total_loss = total_loss / data_source.size(0)

        '''     

        if args.dump_hiddens:
            loss, entropy, hiddens = tl_model.evaluate(data_source, eos_tokens, args.dump_hiddens)
            dump_hiddens(hiddens, 'hiddens_' + str(epoch))
        else:
            loss, entropy = tl_model.evaluate(data_source, eos_tokens)

        #loss = loss.item()
        if args.dump_words:
            W = tl_model.rnn.module.weight_ih_l0.detach()
            dump_words(torch.nn.functional.linear(tl_model.encoder.weight.detach(), W).detach().cpu().numpy(), 'words_xW_' + str(epoch))
            dump_words(tl_model.encoder.weight.detach().cpu().numpy(), 'words_' + str(epoch))

        if not args.dump_entropy is None:
            dump(entropy, args.dump_entropy + str(epoch))

        '''
        return total_loss


    def train():

        # Turn on training mode which enables dropout.
        total_loss, avrg_loss = 0, 0
        start_time = time.time()
        ntokens = len(corpus.dictionary)
        batch, i = 0, 0

        # need a hidden state for tl and one for mos
        h_tl = tl_model.init_hidden(args.batch_size)
        h_mos = mos_model.init_hidden(args.batch_size)
        while i < train_data.size(0)-1:

            # get seq len from batch
            seq_len = seq_lens[batch] - 1

            # adapt learning rate
            lr2 = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
            tl_model.train()

            # get data and binary map
            data = get_batch(train_data, i, args, seq_len=seq_len)
            binary = get_batch(binary_data, i, args, seq_len=seq_len)

            # evaluate mos on data
            h_mos = repackage_hidden(h_mos)
            mos_data = data.clone(); mos_data[mos_data >= 21] = 0 # ugly fix!!!!
            log_prob, h_mos = mos_model(mos_data, h_mos)

            # get probability ranks from mos probability
            _, argsort = torch.sort(log_prob, descending=True)
            argsort = argsort.view(-1, 20)#10000)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset. 
            h_tl = tl_model.init_hidden(args.batch_size)
            h_tl = repackage_hidden(h_tl)

            optimizer.zero_grad()

            #raw_loss = model.train_crossentropy(data, eos_tokens)
            raw_loss = tl_model(data, binary, h_tl, argsort)
            avrg_loss = avrg_loss + (seq_len+1)*raw_loss.data / train_data.size(0)

            loss = raw_loss
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
            optimizer.step()

            total_loss += loss.data
            optimizer.param_groups[0]['lr'] = lr2
            if batch % args.log_interval == 0:
                cur_loss = total_loss.item() / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                    epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / args.log_interval, cur_loss, np.exp(cur_loss), cur_loss / math.log(2)))
                total_loss = 0
                start_time = time.time()
            ###
            batch += 1
            i += seq_len + 1

            #break

        return avrg_loss #/ train_data.size(0)

    # Loop over epochs.
    lr = args.lr
    best_val_loss = []
    valid_loss = []
    stored_loss = 100000000

    W_norm, U_norm = [], []

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        optimizer = None
        # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
        if args.optimizer == 'sgd':
            if threshold_config.lr > 0. and threshold_config.func == 'dynamic':
                optimizer = torch.optim.SGD([{"params": list(tl_model.rnn.parameters()) + list(tl_model.encoder.parameters()) + list(tl_model.decoder.parameters())},
                                            {"params": list(tl_model.threshold.parameters()), "lr":threshold_config.lr}], lr=args.lr, weight_decay=reg_config.wdecay)
            else:
                optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=reg_config.wdecay)
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=reg_config.wdecay)
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train_loss = train()
            #print(model.encoder.weight)
            #print(model.b_h)                
            # look at singular values of [W, U]
            #W = model.rnn.module.weight_ih_l0.detach()
            #U = model.rnn.module.weight_hh_l0.detach()
            #u, s, v = torch.svd(torch.cat([W, U], 1))
            #print(u, s, v)

            #_, s, _= np.linalg.svd(model.rnn.module.weight_hh_l0.cpu().detach().numpy())
            W_norm.append(tl_model.rnn.module.weight_ih_l0.norm())
            U_norm.append(tl_model.rnn.module.weight_hh_l0.norm())

            #print(W_norm[-1])
            #print(U_norm[-1])
            #dump(model.decoder.bias.cpu().detach().numpy(), 'bias_' + str(epoch) +'.out')
            
            # skip to beginning if not in evaluation mode
            if epoch % args.evaluate_every > 0:
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} |'.format(
                        epoch, (time.time() - epoch_start_time), train_loss))
                print('-' * 89) 
                continue

            # evaluate validation loss 
            if 't0' in optimizer.param_groups[0]:
                tmp = {}
                for prm in tl_model.parameters():
                    #if 'ax' in optimizer.state[prm]:
                    tmp[prm] = prm.data.clone()
                    if 'ax' in optimizer.state[prm]:
                        prm.data = optimizer.state[prm]['ax'].clone()

                val_loss2 = evaluate(val_data, epoch)
                valid_loss.append(val_loss2)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                        epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
                print('-' * 89)

                if val_loss2 < stored_loss:
                    model_save(args.save)
                    print('Saving Averaged!')
                    stored_loss = val_loss2

                for prm in tl_model.parameters():
                    prm.data = tmp[prm].clone()

            else:
                val_loss = evaluate(val_data, epoch, eval_batch_size)
                valid_loss.append(val_loss)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                  epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
                print('-' * 89)

                if val_loss < stored_loss:
                    model_save(args.save)
                    print('Saving model (new best validation)')
                    stored_loss = val_loss

                if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                    print('Switching to ASGD')
                    optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=reg_config.wdecay)

                if epoch in args.when:
                    print('Saving model before learning rate decreased')
                    model_save('{}.e{}'.format(args.save, epoch))
                    print('Dividing learning rate by 10')
                    optimizer.param_groups[0]['lr'] /= 10.

                best_val_loss.append(val_loss)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    model_load(args.save)

    # Run on test data.
    test_loss = evaluate(test_data, args.epochs+1, test_batch_size)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
        test_loss, math.exp(test_loss), test_loss / math.log(2)))
    print('=' * 89)

    dump(np.array(W_norm), 'W_norm')
    dump(np.array(U_norm), 'U_norm')
    dump(tl_model.rnn.module.weight_ih_l0.detach().cpu().numpy(), 'W')
    dump(tl_model.rnn.module.weight_hh_l0.detach().cpu().numpy(), 'U')

    return np.array(valid_loss), test_loss

'''
    ### MAIN ###
'''

valid_loss, test_loss = run(args, rnn_config, reg_config, threshold_config, sample_config, bucket_config)
#args.dist_fn = 'poinc'
#valid_loss, test_loss = run(args)
'''
results = []
l = [10]#, (0., 0.25, 0.), (0., 0., 0.25)]
from collections import namedtuple
for temp in l:
    #args.dump_entropy = None
    args.dump_valloss = None
    #args.lr = li
    #reg_config = RegularizationConfig(reg_config.dropout, reg_config.dropouth, dropouti, dropoute, wdrop, reg_config.alpha, reg_config.beta, reg_config.wdecay)
    #rnn_config = RNNConfig('rnn', 100, 100, temp, None)
    #threshold_config = ThresholdConfig(threshold_config.func, threshold_config.mode, threshold_config.temp, threshold_config.min_radius, li, threshold_config.decrease_radius,  threshold_config.nlayers, threshold_config.nhid, threshold_config.lr)
    valid_loss, test_loss = run(args, rnn_config, reg_config, threshold_config, sample_config)

    results.append(( min(valid_loss), test_loss))

for result in results:
    print(result)
'''
'''
l = [[('adam', 1e-3)],
    [4],
    [4, 8],
    [1, 10],
    [1, 10]]
args.dump_entropy = None
args.dump_valloss = None
import itertools
L = list(itertools.product(*l))
results = []
for (opt, lr), nlayers, nhid, temp, ttemp in L:

    settings = [opt, lr, nlayers, nhid, temp, ttemp]
    args.optimizer = opt
    args.lr = lr
    args.temperature = temp
    args.threshold_temp = ttemp
    args.threshold_nlayers = nlayers
    args.threshold_nhid = nhid

    valid_loss, test_loss = run(args)
    results.append(settings + [valid_loss])

for result in results:
    print(result)
'''
