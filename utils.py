# coding: utf-8
import os
import sys
import math
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import uniform, normal

from data import sentence2id, id2sentence, padding_inputs

from encoder import EncoderRNN
from decoder import DecoderRNN
from discriminator import Discriminator
from ael import ApproximateEmbeddingLayer

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

logger = logging.getLogger("GAN-AEL")

def get_variables(batch, vocab, dec_max_len, use_cuda=True):
    """
    Args:
        - **batch**: (list, list) each list is a batch of variable-length sequence
    Outputs:
        Variables for network
    """
    post_ids = [sentence2id(sent, vocab) for sent in batch[0]]
    # add GO
    response_ids = [[GO_ID] + sentence2id(sent, vocab) for sent in batch[1]]
    reference_ids = [sentence2id(sent, vocab) for sent in batch[1]]


    posts_var, posts_length = padding_inputs(post_ids, None)
    responses_var, responses_length = padding_inputs(response_ids, dec_max_len)
    # add EOS
    references_var, references_length = padding_inputs(reference_ids, dec_max_len, eos=True)

    # sort by post length
    posts_length, perms_idx = posts_length.sort(0, descending=True)
    posts_var = posts_var[perms_idx]
    responses_var = responses_var[perms_idx]
    responses_length = responses_length[perms_idx]
    references_var = references_var[perms_idx]
    references_length = references_length[perms_idx]

    if use_cuda:
        posts_var = posts_var.cuda()
        responses_var = responses_var.cuda()
        references_var = references_var.cuda()
    
    return posts_var, posts_length, responses_var, responses_length, references_var, references_length


def get_seq2seq_loss(batch, vocab, encoder, decoder, loss_fn, args): 
    # variable for network
    posts_var, posts_length, responses_var, responses_length, references_var, references_length = get_variables(batch, vocab, args.dec_max_len, args.use_cuda)
    
    # forward
    _, dec_init_state = encoder(posts_var, inputs_length=posts_length.numpy())
    output, _ = decoder(dec_init_state, responses_var) # [B, T, vocab_size]
    
    loss = loss_fn(output.view(-1, output.size(2)),
                     references_var.view(-1))
    return loss

def mask(var):
    return torch.gt(var, 0).float()


def check_go_embedding(args):
    encoder = EncoderRNN(args.vocab_size, args.embedding_dim, args.hidden_dim, 
                      args.n_layers, args.dropout_p, args.rnn_cell)
    
    encoder.load_state_dict(torch.load(
            os.path.join(args.resume_dir, '%s.encoder.params.pkl' % args.resume_prefix)))
    go_vec = encoder.embedding.weight.cpu().data.numpy()
    assert go_vec.shape == (args.vocab_size, args.embedding_dim)
    return np.mean(go_vec, axis=1), np.var(go_vec, axis=1)

def get_gan_loss(batch, vocab, args, encoder, decoder, discriminator, ael=None, noise_go=False):
    """
    Outputs:
        - **D_loss**
        - **G_loss**
    """
    posts_var, posts_length, responses_var, _, _, _ = get_variables(batch, vocab, args.dec_max_len, args.use_cuda)

    # mask
    masked = mask(responses_var)

    _, dec_init_state = encoder(posts_var, inputs_length=posts_length.numpy())

    # greedy decoding
    outputs = []
    state = dec_init_state
    for i in  range(args.dec_max_len):
        if i == 0:
            if noise_go:
                noise_embedding = Variable(torch.Tensor(1, args.embedding_dim).normal_(-0.02, 0.02), requires_grad=False)
                next_word_embedding = torch.stack([noise_embedding]*posts_var.size(0), 0)
                if args.use_cuda:
                    next_word_embedding = next_word_embedding.cuda() # [b, 1, emb_dim]
            else:
                dec_inp_var = Variable(torch.LongTensor([[GO_ID]]*posts_var.size(0)), requires_grad=False)
                if args.use_cuda:
                    dec_inp_var = dec_inp_var.cuda() # [b, 1]
                next_word_embedding = decoder.embedding(dec_inp_var)
        output, state = decoder.rnn(next_word_embedding, state)
        # output = [b, 1, vocab_size]
        if ael:
            next_word_embedding = ael(output)
        else:
            dec_inp_var = torch.max(output.squeeze(1), dim=1, keepdim=True)[1]
            next_word_embedding = decoder.embedding(dec_inp_var)
        outputs.append(next_word_embedding) # [b, 1, emb_dim]

    fake_responses = torch.cat(outputs, dim=1) # [b, T, emb_dim]
    fake_responses = fake_responses * (masked.unsqueeze(-1).expand_as(fake_responses))
    
    real_responses = decoder.embedding(responses_var)
    embedded_posts = encoder.embedding(posts_var)

    prob_real = discriminator(embedded_posts, real_responses) #[b, 1]
    prob_fake = discriminator(embedded_posts, fake_responses)

    D_loss = -torch.mean(torch.log(prob_real) + torch.log(1. - prob_fake)) # [1,]
    G_loss = torch.mean(torch.log(1. - prob_fake))

    return D_loss, G_loss, prob_real, prob_fake


def early_stopping(eval_losses, diff):
    assert isinstance(diff, float), "early_stopping should be either None or float value. Got {}".format(diff)
    eval_loss_diff = eval_losses[-2] - eval_losses[-1]
    if eval_loss_diff < diff:
        logger.info("Evaluation loss stopped decreased less than {}. Early stopping now.".format(diff))
        return True
    else:
        return False


def save_model(save_dir, prefix, encoder=None, decoder=None, discriminator=None):
    if encoder:
        torch.save(encoder.state_dict(), os.path.join(save_dir, '%s.encoder.params.pkl' % prefix))
    if decoder:
        torch.save(decoder.state_dict(), os.path.join(save_dir, '%s.decoder.params.pkl' % prefix))
    if discriminator:
        torch.save(discriminator.state_dict(), os.path.join(save_dir, '%s.discriminator.params.pkl' % prefix))
    logger.info('Save model (prefix = %s) in %s' % (prefix, save_dir))


def reload_model(reload_dir, prefix, encoder, decoder, discriminator=None):
    if os.path.exists(os.path.join(reload_dir, '%s.encoder.params.pkl' % prefix)):
        if encoder:
            encoder.load_state_dict(torch.load(
            os.path.join(reload_dir, '%s.encoder.params.pkl' % prefix)))
        if decoder:
            decoder.load_state_dict(torch.load(
            os.path.join(reload_dir, '%s.decoder.params.pkl' % prefix)))
        if discriminator:
            discriminator.load_state_dict(torch.load(
            os.path.join(reload_dir, '%s.discriminator.params.pkl' % prefix)))
        logger.info("Loading parameters from %s in prefix %s" % (reload_dir, prefix))
    else:
        logger.info("No stored model to load from %s in prefix %s" % (reload_dir, prefix))
        sys.exit()


def eval_model(valid_loader, vocab, encoder, decoder, loss_fn, args):
    logger.info('---------------------validating--------------------------')
    
    #loss_trace = []
    total_loss = 0.0
    step = 0
    #for batch in valid_loader: 
    while True:
        try:
            batch = valid_loader.next()
            step = step + 1
        except:
            break
        loss = get_seq2seq_loss(batch, vocab, encoder, decoder, loss_fn, args)
        total_loss +=  loss.cpu().data.numpy()[0]
        #loss_trace.append(loss)

    #loss_trace = torch.cat(loss_trace, dim=0)
    ave_loss = total_loss / step#torch.sum(loss_trace).cpu().data.numpy()[0]/loss_trace.size()
    #ave_loss = torch.sum(loss_trace).cpu().data.numpy()[0]/loss_trace.size()
    ave_ppl = math.exp(ave_loss)
    logger.info('average valid perplexity %.2f' % ave_ppl)
    return ave_ppl

def build_seq2seq(args):
    encoder = EncoderRNN(args.vocab_size, args.embedding_dim, args.hidden_dim, 
                      args.n_layers, args.dropout_p, args.rnn_cell)
    decoder = DecoderRNN(args.dec_max_len, encoder.embedding, args.vocab_size, 
                      args.embedding_dim, 2*args.hidden_dim, encoder.n_layers,
                      args.dropout_p, args.rnn_cell, use_attention=False)
    
    if args.use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    
    return encoder, decoder

def build_gan(args, adv_args):
    encoder = EncoderRNN(args.vocab_size, args.embedding_dim, args.hidden_dim, 
                      args.n_layers, args.dropout_p, args.rnn_cell)
    decoder = DecoderRNN(args.dec_max_len, encoder.embedding, args.vocab_size, 
                      args.embedding_dim, 2*args.hidden_dim, encoder.n_layers, 
                      args.dropout_p, args.rnn_cell, use_attention=False)
    ael = ApproximateEmbeddingLayer(decoder.embedding)
    discriminator = Discriminator(args.embedding_dim, adv_args.filter_num, eval(adv_args.filter_sizes)) 

    if args.use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        ael = ael.cuda()
        discriminator = discriminator.cuda()
    
    return encoder, decoder, ael, discriminator


def common_opt(parser):
    parser.add_argument('--data_name', help='the name of dataset such as weibo',
                        type=str, required=True)
    parser.add_argument('--mode', '-m', choices=('pretrain', 'adversarial', 'inference'),
                        type=str, required=True)
    parser.add_argument('--exp_dir', type=str, default=None)
    
    # train
    parser.add_argument('--batch_size', '-b', type=int, default=180)
    parser.add_argument('--num_epoch', '-ne', type=int, default=10)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--use_cuda', default=True)
    parser.add_argument('--early_stopping', type=float, default=0.1)

    # resume
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_dir', type=str)
    parser.add_argument('--resume_prefix', type=str)

    # model
    parser.add_argument('--vocab_size', '-vs', type=int, default=50000)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout_p', '-dp', type=float, default=0.5)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--rnn_cell', type=str, default='gru')
    parser.add_argument('--dec_max_len', type=int, default=15)

def predict_opt(parser):
    parser.add_argument('--test_query_file', type=str, required=True)
    parser.add_argument('--load_path', type=str, required=True)
    # TODO: load epoch -> load best model
    parser.add_argument('--load_prefix', type=str, required=True)
    
    parser.add_argument('--output_file', '-o', type=str, default=None)


def data_opt(parser):
    parser.add_argument('--train_query_file', '-tqf', type=str, required=True)
    parser.add_argument('--train_response_file', '-trf', type=str, required=True)
    parser.add_argument('--valid_query_file', '-vqf', type=str, required=True)
    parser.add_argument('--valid_response_file', '-vrf', type=str, required=True)
    parser.add_argument('--vocab_file', '-vf', type=str, default='')

def seq2seq_opt(parser):
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)


def adversarial_opt(parser):
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--load_path', type=str, required=True)
    # TODO: load epoch -> load best model
    parser.add_argument('--load_prefix', type=str, required=True)
    
    parser.add_argument('--training_ratio', type=int, default=2)
    parser.add_argument('--g_learning_rate', '-glr', type=float, default=0.00001)
    parser.add_argument('--d_learning_rate', '-dlr', type=float, default=0.00001)
    parser.add_argument('--d_pretrain_learning_rate', '-dplr', type=float, default=0.001)
    
    parser.add_argument('--filter_num', type=int, default=30)
    parser.add_argument('--filter_sizes', type=str, default='[1,2,3,4]')
     
def make_link(src_path, dst):
    dst_path = os.path.join(os.path.dirname(os.path.abspath(src_path)), dst)
    os.symlink(os.path.basename(src_path), dst_path)


