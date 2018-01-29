# coding:utf-8
"""
USEAGE:

    CUDA_VISIBLE_DEVICES=7 python inference.py -i=dataset/weibo/stc_weibo_train_post -p=exp/weibo/pretrain/2018-01-29-11-43-34/ -e=1
"""
import os
import pickle
import argparse
from data import batcher, load_vocab, sentence2id, id2sentence
from toy import reload_model

import torch
import torch.nn as nn

from encoder import EncoderRNN, padding_inputs
from generator import Generator

from utils import SYM_PAD
USE_CUDA = True

def predict(test_post_file, vocab, rev_vocab, word_embeddings, encoder, generator, algo='greedy', output_file=None):
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
        if USE_CUDA:
            posts_var = posts_var.cuda()

        embedded_post = word_embeddings(posts_var)
        _, dec_init_state = encoder(embedded_post, input_lengths=posts_length.numpy())
        log_softmax_outputs = generator.inference(dec_init_state, word_embeddings) # [B, T, vocab_size]
        
        if algo == 'greedy':
            # or 'beam'
            _, results = torch.max(log_softmax_outputs, dim=2) # [B, T]
        else:
            pass
        
        response_sentence = id2sentence(results.cpu().data.numpy().reshape(-1).tolist(), rev_vocab)
        if not output_file:
            print('*******************************************************')
            print "post:" + ''.join(post_sentence[0])
            print "response:" + ''.join(response_sentence)
        else:
            fo.write("post: %s\nresponse: %s\n\n" % (
                ''.join(post_sentence[0]), 
                '',join(response_sentence)))
    if output_file:
        fo.close()

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--test_query_file', '-i', type=str, required=True)
    argparser.add_argument('--load_path', '-p', type=str, required=True)
    # TODO: load epoch -> load best model
    argparser.add_argument('--load_epoch', '-e', type=int, required=True)
    
    argparser.add_argument('--output_file', '-o', type=str)
    argparser.add_argument('--dec_algorithm', '-algo', type=str, default='greedy')
    
    new_args = argparser.parse_args()
   
    arg_file = os.path.join(new_args.load_path, 'args.pkl')
    if not os.path.exists(arg_file):
        raise RuntimeError('No default arguments file to load')
    f = open(arg_file, 'rb')
    args = pickle.load(f)
    f.close()

    if args.use_cuda:
        USE_CUDA = True
    
    vocab, rev_vocab = load_vocab(args.vocab_file, max_vocab=args.max_vocab_size)
    vocab_size = len(vocab)

    word_embeddings = nn.Embedding(vocab_size, args.emb_dim, padding_idx=SYM_PAD)
    E = EncoderRNN(vocab_size, args.emb_dim, args.hidden_dim, args.n_layers, bidirectional=True, variable_lengths=True)
    G = Generator(vocab_size, args.response_max_len, args.emb_dim, 2*args.hidden_dim, args.n_layers)
    
    if USE_CUDA:
        word_embeddings.cuda()
        E.cuda()
        G.cuda()

    reload_model(word_embeddings, E, G, new_args.load_path, new_args.load_epoch)
    
    predict(new_args.test_query_file,
            vocab, rev_vocab,
            word_embeddings, E, G,
            new_args.dec_algorithm,
            new_args.output_file)

if __name__ == '__main__':
    main()

