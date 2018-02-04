"""
USAGE:
    CUDA_VISIBLE_DEVICES=4 python adversarial.py --load_path=exp/weibo/pretrain/2018-02-02-13-37-43/ --load_prefix=best --batch_size=80 --d_pretrain_learning_rate=0.0001


"""

import os
import time
import utils
import pickle
import argparse
import logging
import tqdm_logging
from utils import build_gan, save_model, reload_model, get_gan_loss

import torch

from data import load_vocab, batcher


logger = logging.getLogger("GAN-AEL")


def d_pretraining(args, adv_args, vocab, encoder, decoder, discriminator):
    num_epoch = 2
    logger.info('----------------------------------')
    logger.info('Pretraining discriminator (epoch = %d)' % num_epoch)
    logger.info('----------------------------------')
    
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=adv_args.d_pretrain_learning_rate)
    
    for e in range(num_epoch):
        logger.info('---------------------training--------------------------')
        logger.info("Epoch: %d/%d" % (e, num_epoch))
        train_batcher = batcher(args.batch_size, args.train_query_file, args.train_response_file)
        step = 0
        cur_time = time.time() 
        while True:
            try:
                batch = train_batcher.next()
            except StopIteration:
                save_model(args.exp_dir, 'dis_pre%d' % (e+1), None, None, discriminator) 
                break
            D_loss, G_loss, prob_real, prob_fake = get_gan_loss(batch, vocab, args.dec_max_len, args.use_cuda,
                                          encoder, decoder, discriminator, None)
            
            opt_D.zero_grad()
            D_loss.backward()
            opt_D.step()
            
            if step % args.print_every == 0:
                logger.info('Step %5d: D loss=%.2f D accuracy=%.2f (%.1f iters/sec)' % (
                    step, 
                    D_loss.cpu().data.numpy()[0],
                    prob_real.cpu().data.numpy().mean(), 
                    args.print_every/(time.time()-cur_time)))
                cur_time = time.time()
            step = step + 1
 

def adv_training(args, adv_args, vocab, encoder, decoder, discriminator, ael):
    logger.info('----------------------------------')
    logger.info('Adversarial-training discriminator & generator')
    logger.info('----------------------------------')

    # define optimizer
    opt_G = torch.optim.Adam(decoder.rnn.parameters(), lr=adv_args.g_learning_rate)
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=adv_args.d_learning_rate)
    
    for e in range(args.num_epoch):
        logger.info('---------------------training--------------------------')
        logger.info("Epoch: %d/%d" % (e, args.num_epoch))
        
        # load data
        train_batcher = batcher(args.batch_size, args.train_query_file, args.train_response_file)
        
        cur_time = time.time() 
        step = 0
        while True:
            try:
                batch = train_batcher.next()
            except:
                save_model(args.exp_dir, 'adv%d' % (e+1), encoder, decoder, discriminator)
                break
            # ael is necessary because of adversarial
            D_loss, G_loss, prob_real, prob_fake = get_gan_loss(batch, vocab, args.dec_max_len, args.use_cuda, encoder, decoder, discriminator, ael)
            
            if step % adv_args.training_ratio == 0:
                opt_D.zero_grad()
                D_loss.backward(retain_graph=True)
                opt_D.step()
            
            opt_G.zero_grad()
            G_loss.backward()
            opt_G.step()

            if step % args.print_every == 0:
                logger.info('Step %5d: D accuracy=%.2f (%.2f) (0.5 for D to converge) D score=%.2f (-1.38 for G to converge) (%.1f iters/sec)' % (
                    step, 
                    prob_real.cpu().data.numpy().mean(), 
                    prob_fake.cpu().data.numpy().mean(), 
                    -D_loss.cpu().data.numpy()[0], 
                    args.print_every/(time.time()-cur_time)))
                cur_time = time.time()
            step = step + 1
            

def run(args, adv_args):
    vocab, _ = load_vocab(args.vocab_file, max_vocab=args.vocab_size)
    # build
    encoder, decoder, ael, discriminator = build_gan(args, adv_args)
    # load params
    reload_model(adv_args.load_path, adv_args.load_prefix, encoder, decoder, None)
    
    # training
    d_pretraining(args, adv_args, vocab, encoder, decoder, discriminator)
    adv_training(args, adv_args, vocab, encoder, decoder, discriminator, ael)

def main():
    parser = argparse.ArgumentParser('adversarial')
    utils.adversarial_opt(parser)
    adv_args = parser.parse_args()

    # load default args
    arg_file = os.path.join(adv_args.load_path, 'args.pkl')
    if not os.path.exists(arg_file):
        raise RuntimeError('No default arguments file to load')
    with open(arg_file, 'rb') as f:
        args = pickle.load(f)
    
    args.mode = 'adversarial'
    args.batch_size = adv_args.batch_size

    # set up the output directory
    exp_dir = os.path.join('exp', args.data_name, args.mode, time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(exp_dir)
    args.exp_dir = exp_dir

    # store args
    with open(os.path.join(exp_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)
        pickle.dump(adv_args, f)
    
    # set up the logger
    tqdm_logging.config(logger, os.path.join(exp_dir, '%s.log' % args.mode), 
                        mode='w', silent=False, debug=True)

    logger.info('Args:')
    logger.info(str(args))
    logger.info('Adv Args:')
    logger.info(str(adv_args))

    run(args, adv_args)


if __name__ == '__main__':
    main()


