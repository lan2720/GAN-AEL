# coding:utf-8

import os
import time
import pickle
import argparse
import logging
import tqdm_logging

from utils import SYM_PAD
from data import batcher, load_vocab, padding_inputs, sentence2id
from encoder import EncoderRNN
from generator import Generator
from discriminator import Discriminator
from toy import save_model, reload_model


import torch
import torch.nn as nn


def adversarial():
    # user the root logger
    logger = logging.getLogger("lan2720")
    
    argparser = argparse.ArgumentParser(add_help=False)
    argparser.add_argument('--load_path', '-p', type=str, required=True)
    # TODO: load best
    argparser.add_argument('--load_epoch', '-e', type=int, required=True)
    argparser.add_argument('--filter_num', type=int, required=True)
    argparser.add_argument('--filter_sizes', type=str, required=True)
    argparser.add_argument('--training_ratio', type=int, default=2)

    
    # new arguments used in adversarial
    new_args = argparser.parse_args()
    
    # load default arguments
    default_arg_file = os.path.join(new_args.load_path, 'args.pkl')
    if not os.path.exists(default_arg_file):
        raise RuntimeError('No default argument file in %s' % new_args.load_path)
    else:
        with open(default_arg_file, 'rb') as f:
            args = pickle.load(f)
    
    args.mode = 'adversarial'
    #args.d_learning_rate  = 0.0001
    args.print_every = 1

    # set up the output directory
    exp_dirname = os.path.join(args.exp_dir, args.mode, time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(exp_dirname)

    # set up the logger
    tqdm_logging.config(logger, os.path.join(exp_dirname, 'adversarial.log'), 
                        mode='w', silent=False, debug=True)

    # load vocabulary
    vocab, rev_vocab = load_vocab(args.vocab_file, max_vocab=args.max_vocab_size)

    vocab_size = len(vocab)

    word_embeddings = nn.Embedding(vocab_size, args.emb_dim, padding_idx=SYM_PAD)
    E = EncoderRNN(vocab_size, args.emb_dim, args.hidden_dim, args.n_layers, args.dropout_rate, bidirectional=True, variable_lengths=True)
    G = Generator(vocab_size, args.response_max_len, args.emb_dim, 2*args.hidden_dim, args.n_layers, dropout_p=args.dropout_rate)
    D = Discriminator(args.emb_dim, new_args.filter_num, eval(new_args.filter_sizes))
    
    if args.use_cuda:
        word_embeddings.cuda()
        E.cuda()
        G.cuda()
        D.cuda()

    # define optimizer
    G_params = list(word_embeddings.parameters()) + list(E.parameters()) + list(G.parameters())
    opt_G = torch.optim.Adam(G_params, lr=args.g_learning_rate)
    opt_D = torch.optim.Adam(D.parameters(), lr=args.d_learning_rate)
    
    logger.info('----------------------------------')
    logger.info('Adversarial a neural conversation model')
    logger.info('----------------------------------')

    logger.info('Args:')
    logger.info(str(args))
    
    logger.info('Vocabulary from ' + args.vocab_file)
    logger.info('vocabulary size: %d' % vocab_size)
    logger.info('Loading text data from ' + args.train_query_file + ' and ' + args.train_response_file)
   
    
    reload_model(new_args.load_path, new_args.load_epoch, word_embeddings, E, G)
    #    start_epoch = args.resume_epoch + 1
    #else:
    #    start_epoch = 0

    # dump args
    with open(os.path.join(exp_dirname, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)


    # TODO: num_epoch is old one
    for e in range(args.num_epoch):
        train_data_generator = batcher(args.batch_size, args.train_query_file, args.train_response_file)
        logger.info("Epoch: %d/%d" % (e, args.num_epoch))
        step = 0
        cur_time = time.time() 
        while True:
            try:
                post_sentences, response_sentences = train_data_generator.next()
            except StopIteration:
                # save model
                save_model(exp_dirname, e, word_embeddings, E, G, D) 
                ## evaluation
                #eval(args.valid_query_file, args.valid_response_file, args.batch_size, 
                #        word_embeddings, E, G, loss_func, args.use_cuda, vocab, args.response_max_len)
                break
            
            # prepare data
            post_ids = [sentence2id(sent, vocab) for sent in post_sentences]
            response_ids = [sentence2id(sent, vocab) for sent in response_sentences]
            posts_var, posts_length = padding_inputs(post_ids, None)
            responses_var, responses_length = padding_inputs(response_ids, args.response_max_len)
            # sort by post length
            posts_length, perms_idx = posts_length.sort(0, descending=True)
            posts_var = posts_var[perms_idx]
            responses_var = responses_var[perms_idx]
            responses_length = responses_length[perms_idx]

            if args.use_cuda:
                posts_var = posts_var.cuda()
                responses_var = responses_var.cuda()

            embedded_post = word_embeddings(posts_var)
            real_responses = word_embeddings(responses_var)

            # forward
            _, dec_init_state = E(embedded_post, input_lengths=posts_length.numpy())
            fake_responses = G(dec_init_state, word_embeddings) # [B, T, emb_size]

            prob_real = D(embedded_post, real_responses)
            prob_fake = D(embedded_post, fake_responses)
        
            # loss
            D_loss = - torch.mean(torch.log(prob_real) + torch.log(1. - prob_fake)) 
            G_loss = torch.mean(torch.log(1. - prob_fake))
            
            if step % new_args.training_ratio == 0:
                opt_D.zero_grad()
                D_loss.backward(retain_graph=True)
                opt_D.step()
            
            opt_G.zero_grad()
            G_loss.backward()
            opt_G.step()
            
            if step % args.print_every == 0:
                logger.info('Step %5d: D accuracy=%.2f (0.5 for D to converge) D score=%.2f (-1.38 for G to converge) (%.1f iters/sec)' % (
                    step, 
                    prob_real.cpu().data.numpy().mean(), 
                    -D_loss.cpu().data.numpy()[0], 
                    args.print_every/(time.time()-cur_time)))
                cur_time = time.time()
            step = step + 1

if __name__ == '__main__':
    adversarial()
