"""
USAGE:
    CUDA_VISIBLE_DEVICES=7 python pretrain.py -tqf=dataset/weibo/stc_weibo_train_post -trf=dataset/weibo/stc_weibo_train_response  -vqf=dataset/weibo/stc_weibo_valid_post -vrf=dataset/weibo/stc_weibo_valid_response -vf=vocab.707749 --data_name=weibo -m=pretrain -b=128  --resume --resume_dir=exp/weibo/pretrain/2018-02-01-11-40-16/ --resume_epoch=1
"""

import os
import time
import math
import utils
import pickle
import argparse
import logging
import tqdm_logging
from utils import build_seq2seq, eval_model, save_model, reload_model, get_seq2seq_loss, early_stopping, make_link

import torch
import torch.nn as nn

from data import load_vocab, batcher



logger = logging.getLogger("GAN-AEL")

def training(args, encoder, decoder):
    vocab, rev_vocab = load_vocab(args.vocab_file, max_vocab=args.vocab_size)

    logger.info('----------------------------------')
    logger.info('Pre-train a neural conversation model')
    logger.info('----------------------------------')

    logger.info('Args:')
    logger.info(str(args))
    
    logger.info('Vocabulary from ' + args.vocab_file)
    logger.info('Loading train data from ' + args.train_query_file + ' and ' + args.train_response_file)
    logger.info('Loading valid data from ' + args.valid_query_file + ' and ' + args.valid_response_file)
 
    # resume training from other experiment
    if args.resume:
        reload_model(args.resume_dir, args.resume_prefix, encoder, decoder)

    loss_func = nn.CrossEntropyLoss(ignore_index=utils.PAD_ID) 
    opt = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()),
                           lr=args.learning_rate)

    valid_batcher = batcher(args.batch_size, args.valid_query_file, args.valid_response_file)

    # training
    valid_ppl_trace = []
    best_valid_ppl = eval_model(valid_batcher, vocab, encoder, decoder, loss_func, args)
    valid_ppl_trace.append(best_valid_ppl)
    
    for e in range(args.num_epoch):
        logger.info('---------------------training--------------------------')
        logger.info("Epoch: %d/%d" % (e, args.num_epoch))
        
        # load data
        train_batcher = batcher(args.batch_size, args.train_query_file, args.train_response_file)
        valid_batcher = batcher(args.batch_size, args.valid_query_file, args.valid_response_file)
        
        total_loss = 0.0
        total_case = 0
        cur_time = time.time() 
        step = 0
        while True:
            try:
                batch = train_batcher.next()
            except:
                break
            loss = get_seq2seq_loss(batch, vocab, encoder, decoder, loss_func, args)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss
            total_case += len(batch[0])

            if step % args.print_every == 0:
                #loss_trace = torch.cat(loss_trace, dim=0)
                ave_loss = total_loss.cpu().data.numpy()[0]
                if step > 0:
                    ave_loss = ave_loss/args.print_every
                logger.info('batch %d: average train perplexity %.2f (%.1f case/sec)' % (
                             step, math.exp(ave_loss), 
                             total_case/(time.time()-cur_time)))
                total_loss = 0.0
                total_case = 0
                cur_time = time.time()
            step = step + 1
        save_model(args.exp_dir, 's2s%d' % (e+1), encoder, decoder, None)
        cur_valid_ppl = eval_model(valid_batcher, vocab, encoder, decoder, loss_func, args)
        if cur_valid_ppl < best_valid_ppl:
            make_link(os.path.join(args.exp_dir,
                     's2s%d.encoder.params.pkl' % (e+1)),
                     'best.encoder.params.pkl')
            make_link(os.path.join(args.exp_dir,
                     's2s%d.decoder.params.pkl' % (e+1)),
                     'best.decoder.params.pkl')
        # early stop
        valid_ppl_trace.append(cur_valid_ppl)
        valid_ppl_trace = valid_ppl_trace[-5:]
        if early_stopping(valid_ppl_trace, args.early_stopping):
            break
            

def run(args):
    encoder, decoder = build_seq2seq(args)
    training(args, encoder, decoder)


def main():
    parser = argparse.ArgumentParser('pretrain')
    utils.common_opt(parser)
    utils.data_opt(parser)
    utils.seq2seq_opt(parser)
    args = parser.parse_args()

    # set up the output directory
    exp_dir = os.path.join('exp', args.data_name, args.mode, time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(exp_dir)
    args.exp_dir = exp_dir

    # store args
    with open(os.path.join(exp_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # set up the logger
    tqdm_logging.config(logger, os.path.join(exp_dir, '%s.log' % args.mode), 
                        mode='w', silent=False, debug=True)

    run(args)


if __name__ == '__main__':
    main()


