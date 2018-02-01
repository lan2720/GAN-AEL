# coding:utf-8
"""
USEAGE:
    CUDA_VISIBLE_DEVICES=5 python inference.py --test_query_file=dataset/weibo/stc_weibo_valid_post --load_path=exp/weibo/pretrain/2018-02-01-11-40-16/  --load_epoch=0
"""
import os
import pickle
import argparse
import logging
import numpy as np


import utils
import torch
import torch.nn as nn
from torch.autograd import Variable

from encoder import EncoderRNN
from decoder import DecoderRNN

from utils import reload_model, build_seq2seq
from data import batcher, load_vocab, sentence2id, id2sentence, padding_inputs

logger = logging.getLogger('GAN-AEL')


def predict(test_post_file, vocab, rev_vocab, encoder, decoder, args, output_file=None):
    # data generator
    test_data_generator = batcher(1, test_post_file, response_file=None) 
   
    if output_file:
        fo = open(output_file, 'wb')

    while True:
        try:
            post_sentence = test_data_generator.next()
        except StopIteration:
            logger.info('---------------------finish-------------------------')
            break
        
        post_ids = [sentence2id(sent, vocab) for sent in post_sentence]
        posts_var, posts_length = padding_inputs(post_ids, None)
        
        if args.use_cuda:
            posts_var = posts_var.cuda()

        _, dec_init_state = encoder(posts_var, inputs_length=posts_length.numpy())
        hyps, _ = beam_search(dec_init_state, decoder, args, beam=5, penalty=1.0, nbest=1)
        results = []
        for h in hyps:
            results.append(id2sentence(h[0], rev_vocab))

        print('*******************************************************')
        print("post:" + ' '.join(post_sentence[0]).encode('utf-8'))
        print("response:\n" + '\n'.join([' '.join(r) for r in results]).encode('utf-8'))
        print('')

    if output_file:
        fo.close()



def beam_search(dec_init_state, decoder, args, beam=5, penalty=1.0, nbest=1):
    """
    the code is referred to: 
    https://github.com/dialogtekgeek/DSTC6-End-to-End-Conversation-Modeling/blob/master/ChatbotBaseline/tools/seq2seq_model.py
    """
    go_i = Variable(torch.LongTensor([[utils.GO_ID]]), requires_grad=False)
    eos_i = Variable(torch.LongTensor([[utils.EOS_ID]]), requires_grad=False)
    if args.use_cuda:
        go_i = go_i.cuda()
        eos_i = eos_i.cuda()

    ds = decoder.update((None, dec_init_state), go_i)
    hyplist = [([], 0., ds)]
    best_state = None
    comp_hyplist = []
    for l in range(args.dec_max_len):
        new_hyplist = []
        argmin = 0
        for out, lp, st in hyplist:
            logp = decoder.predict(st).squeeze()
            #[vocab_size,]
            lp_vec = logp.cpu().data.numpy() + lp
            if l > 0:
                new_lp = lp_vec[utils.EOS_ID] + penalty*(len(out)+1)
                new_st = decoder.update(st, eos_i)
                comp_hyplist.append((out, new_lp))
                if best_state is None or best_state[0] < new_lp:
                    best_state = (new_lp, new_st)
            
            for o in np.argsort(lp_vec)[::-1]:
                if o == utils.UNK_ID or o == utils.EOS_ID:
                    continue
                new_lp = lp_vec[o]
                o_var = Variable(torch.LongTensor([[o]]), requires_grad=False)
                if args.use_cuda:
                    o_var = o_var.cuda()
                if len(new_hyplist) == beam:
                    if new_hyplist[argmin][1] <  new_lp:
                        new_st = decoder.update(st, o_var)
                        new_hyplist[argmin] = (out+[o], new_lp, new_st)
                        argmin = min(enumerate(new_hyplist), key=lambda h:h[1][1])[0]
                    else:
                        break
                else:
                    new_st = decoder.update(st, o_var)
                    new_hyplist.append((out+[o], new_lp, new_st))
                    if len(new_hyplist) == beam:
                        argmin = min(enumerate(new_hyplist), key=lambda h:h[1][1])[0]
        hyplist = new_hyplist
    
    if len(comp_hyplist):
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:nbest]
        return maxhyps, best_state[1]
    else:
        return [([], 0)], None


def main():
    parser = argparse.ArgumentParser('predict')
    utils.predict_opt(parser)
    pred_args = parser.parse_args()
    
    # load default args
    arg_file = os.path.join(pred_args.load_path, 'args.pkl')
    if not os.path.exists(arg_file):
        raise RuntimeError('No default arguments file to load')
    with open(arg_file, 'rb') as f:
        args = pickle.load(f)
    
    vocab, rev_vocab = load_vocab(args.vocab_file, max_vocab=args.vocab_size)

    encoder, decoder = build_seq2seq(args) 
    
    reload_model(pred_args.load_path, pred_args.load_epoch, encoder, decoder)
    
    predict(pred_args.test_query_file,
            vocab, rev_vocab,
            encoder, decoder,
            args, pred_args.output_file)

if __name__ == '__main__':
    main()

