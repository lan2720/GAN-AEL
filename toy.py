# coding:utf-8

import os
import sys
import math
import time
import pickle
import logging
import tqdm_logging

from utils import SYM_PAD, SYM_GO, SYM_EOS
from data import batcher, build_vocab, load_vocab, sentence2id, id2sentence

from discriminator import Discriminator
from generator import Generator
from encoder import EncoderRNN, padding_inputs

import torch
import torch.nn as nn
from torch.autograd import Variable

import argparse



# user the root logger
logger = logging.getLogger("lan2720")


def format_arguments(args):
    s = []
    for k, v in sorted(vars(args).items(), key=lambda i: i[0]):
        s.append(k + '=' + (str(v) if v != None else ''))
    return '\n'.join(s)

def mask(x):
    """
    返回x的mask矩阵，即x中为0的部分，全部为0，非0的部分，全部为1
    """
    return torch.gt(x, 0).float()

def eval(valid_query_file, valid_response_file, batch_size,
            word_embeddings, E, G, 
            loss_func, use_cuda, 
            vocab, response_max_len):
    logger.info('---------------------validating--------------------------')
    logger.info('Loading valid data from %s and %s' % (valid_query_file, valid_response_file))
    
    valid_data_generator = batcher(batch_size, valid_query_file, valid_response_file)
    
    sum_loss = 0.0
    valid_char_num = 0
    example_num = 0
    while True:
        try:
            post_sentences, response_sentences = valid_data_generator.next()
        except StopIteration:
            # one epoch finish
            logger.info('---------------------finish-------------------------')
            break

        post_ids = [sentence2id(sent, vocab) for sent in post_sentences]
        response_ids = [sentence2id(sent, vocab) for sent in response_sentences]
        posts_var, posts_length = padding_inputs(post_ids, None)
        responses_var, responses_length = padding_inputs(response_ids, response_max_len)
        
        # sort by post length
        posts_length, perms_idx = posts_length.sort(0, descending=True)
        posts_var = posts_var[perms_idx]
        responses_var = responses_var[perms_idx]
        responses_length = responses_length[perms_idx]

        if use_cuda:
            posts_var = posts_var.cuda()
            responses_var = responses_var.cuda()

        embedded_post = word_embeddings(posts_var)
        _, dec_init_state = E(embedded_post, input_lengths=posts_length.numpy())
        log_softmax_outputs = G.inference(dec_init_state, word_embeddings) # [B, T, vocab_size]
        
        outputs = log_softmax_outputs.view(-1, len(vocab))
        mask_pos = mask(responses_var).view(-1).unsqueeze(-1)
        masked_output = outputs*(mask_pos.expand_as(outputs))
        loss = loss_func(masked_output, responses_var.view(-1))

        sum_loss += loss.cpu().data.numpy()[0]
        example_num += posts_var.size(0)
        valid_char_num += torch.sum(mask_pos).cpu().data.numpy()[0]
    
    logger.info('Valid Loss (per case): %.2f Valid Perplexity (per word): %.2f' % (sum_loss/example_num, math.exp(sum_loss/valid_char_num)))


def save_model(word_embeddings, encoder, generator, save_dir, epoch):
    torch.save(word_embeddings.state_dict(), os.path.join(save_dir, 'epoch%d.word_embeddings.params.pkl' % epoch))
    torch.save(encoder.state_dict(), os.path.join(save_dir, 'epoch%d.encoder.params.pkl' % epoch))
    torch.save(generator.state_dict(), os.path.join(save_dir, 'epoch%d.generator.params.pkl' % epoch))
    logger.info('Save model (epoch = %d) in %s' % (epoch, save_dir))
    

def reload_model(word_embeddings, encoder, generator, reload_dir, epoch):
    if os.path.exists(reload_dir):
        word_embeddings.load_state_dict(torch.load(
            os.path.join(reload_dir, 'epoch%d.word_embeddings.params.pkl' % epoch)))
        encoder.load_state_dict(torch.load(
            os.path.join(reload_dir, 'epoch%d.encoder.params.pkl' % epoch)))
        generator.load_state_dict(torch.load(
            os.path.join(reload_dir, 'epoch%d.generator.params.pkl' % epoch)))
        logger.info("Loading parameters from %s in epoch %d" % (reload_dir, epoch))
    else:
        raise RuntimeError("No stored model to load from %s" % reload_dir)


def pretrain():
    # Parse command line arguments
    argparser = argparse.ArgumentParser()

    # train
    argparser.add_argument('--mode', '-m', choices=('pretrain', 'adversarial', 'inference'),
                            type=str, required=True)
    argparser.add_argument('--batch_size', '-b', type=int, default=168)
    argparser.add_argument('--num_epoch', '-e', type=int, default=10)
    argparser.add_argument('--print_every', type=int, default=100)
    argparser.add_argument('--use_cuda', default=True)
    argparser.add_argument('--g_learning_rate', '-glr', type=float, default=0.001)
    argparser.add_argument('--d_learning_rate', '-dlr', type=float, default=0.001)

    # resume
    argparser.add_argument('--resume', action='store_true', dest='resume')
    argparser.add_argument('--resume_dir', type=str)
    argparser.add_argument('--resume_epoch', type=int)

    # save
    argparser.add_argument('--exp_dir', type=str, required=True)

    # model
    argparser.add_argument('--emb_dim', type=int, default=128)
    argparser.add_argument('--hidden_dim', type=int, default=256)
    argparser.add_argument('--n_layers', type=int, default=1)
    argparser.add_argument('--response_max_len', type=int, default=15)


    # data
    argparser.add_argument('--train_query_file', '-tqf', type=str, required=True)
    argparser.add_argument('--train_response_file', '-trf', type=str, required=True)
    argparser.add_argument('--valid_query_file', '-vqf', type=str, required=True)
    argparser.add_argument('--valid_response_file', '-vrf', type=str, required=True)
    argparser.add_argument('--vocab_file', '-vf', type=str, default='')
    argparser.add_argument('--max_vocab_size', '-mv', type=int, default=100000)
    
    args = argparser.parse_args()
    
    # set up the output directory
    exp_dirname = os.path.join(args.exp_dir, args.mode, time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(exp_dirname)

    # set up the logger
    tqdm_logging.config(logger, os.path.join(exp_dirname, 'train.log'), 
                        mode='w', silent=False, debug=True)

    if not args.vocab_file:
        logger.info("no vocabulary file")
        build_vocab(args.train_query_file, args.train_response_file, seperated=True)
        sys.exit()
    else:
        vocab, rev_vocab = load_vocab(args.vocab_file, max_vocab=args.max_vocab_size)

    vocab_size = len(vocab)

    word_embeddings = nn.Embedding(vocab_size, args.emb_dim, padding_idx=SYM_PAD)
    E = EncoderRNN(vocab_size, args.emb_dim, args.hidden_dim, args.n_layers, bidirectional=True, variable_lengths=True)
    G = Generator(vocab_size, args.response_max_len, args.emb_dim, 2*args.hidden_dim, args.n_layers)
    
    if args.use_cuda:
        word_embeddings.cuda()
        E.cuda()
        G.cuda()
    

    loss_func = nn.NLLLoss(size_average=False)
    params = list(word_embeddings.parameters()) + list(E.parameters()) + list(G.parameters())
    opt = torch.optim.Adam(params, lr=args.g_learning_rate)
    

    logger.info('----------------------------------')
    logger.info('Pre-train a neural conversation model')
    logger.info('----------------------------------')

    logger.info('Args:')
    logger.info(str(args))
    
    logger.info('Vocabulary from ' + args.vocab_file)
    logger.info('vocabulary size: %d' % vocab_size)
    logger.info('Loading text data from ' + args.train_query_file + ' and ' + args.train_response_file)
   
    # resume training from other experiment
    if args.resume:
        assert args.resume_epoch >= 0, 'If resume training, please assign resume_epoch'
        reload_model(word_embeddings, E, G, args.resume_dir, args.resume_epoch)
        start_epoch = args.resume_epoch + 1
    else:
        start_epoch = 0

    # dump args
    with open(os.path.join(exp_dirname, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)
    

    for e in range(start_epoch, args.num_epoch):
        logger.info('---------------------training--------------------------')
        train_data_generator = batcher(args.batch_size, args.train_query_file, args.train_response_file)
        logger.info("Epoch: %d/%d" % (e, args.num_epoch))
        step = 0
        total_loss = 0.0
        total_valid_char = []
        cur_time = time.time() 
        while True:
            try:
                post_sentences, response_sentences = train_data_generator.next()
            except StopIteration:
                # save model
                save_model(word_embeddings, E, G, exp_dirname, epoch=e) 
                # evaluation
                eval(args.valid_query_file, args.valid_response_file, args.batch_size, 
                        word_embeddings, E, G, loss_func, args.use_cuda, vocab, args.response_max_len)
                break
            

            post_ids = [sentence2id(sent, vocab) for sent in post_sentences]
            response_ids = [sentence2id(sent, vocab) for sent in response_sentences]
            posts_var, posts_length = padding_inputs(post_ids, None)
            responses_var, responses_length = padding_inputs(response_ids, args.response_max_len)
            # sort by post length
            posts_length, perms_idx = posts_length.sort(0, descending=True)
            posts_var = posts_var[perms_idx]
            responses_var = responses_var[perms_idx]
            responses_length = responses_length[perms_idx]

            # 在sentence后面加eos
            references_var = torch.cat([responses_var, Variable(torch.zeros(responses_var.size(0),1).long(), requires_grad=False)], dim=1)

            for idx, length in enumerate(responses_length):
                references_var[idx, length] = SYM_EOS


            # show case
            #for p, r, ref in zip(posts_var.data.numpy()[:10], responses_var.data.numpy()[:10], references_var.data.numpy()[:10]):
            #    print ''.join(id2sentence(p, rev_vocab))
            #    print ''.join(id2sentence(r, rev_vocab))
            #    print ''.join(id2sentence(ref, rev_vocab))
            #    print

            if args.use_cuda:
                posts_var = posts_var.cuda()
                responses_var = responses_var.cuda()
                references_var = references_var.cuda()

            embedded_post = word_embeddings(posts_var)
            embedded_response = word_embeddings(responses_var)

            _, dec_init_state = E(embedded_post, input_lengths=posts_length.numpy())
            log_softmax_outputs = G.supervise(embedded_response, dec_init_state, word_embeddings) # [B, T, vocab_size]
            
            outputs = log_softmax_outputs.view(-1, vocab_size)
            mask_pos = mask(references_var).view(-1).unsqueeze(-1)
            masked_output = outputs*(mask_pos.expand_as(outputs))
            loss = loss_func(masked_output, references_var.view(-1))/(posts_var.size(0))

            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss*(posts_var.size(0))
            total_valid_char.append(mask_pos)

            if step % args.print_every == 0:
                total_loss_val = total_loss.cpu().data.numpy()[0]
                total_valid_char_val = torch.sum(torch.cat(total_valid_char, dim=1)).cpu().data.numpy()[0]
                logger.info('Step %5d: (per word) training perplexity %.2f (%.1f iters/sec)' % (step, math.exp(total_loss_val/total_valid_char_val), args.print_every/(time.time()-cur_time)))
                total_loss = 0.0
                total_valid_char = []
                total_case_num = 0
                cur_time = time.time()
            step = step + 1

if __name__ == '__main__':
    pretrain()


