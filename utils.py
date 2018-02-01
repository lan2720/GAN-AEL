# coding: utf-8
import os
import sys
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import uniform, normal

from data import sentence2id, id2sentence, padding_inputs

from encoder import EncoderRNN
from decoder import DecoderRNN

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

logger = logging.getLogger("GAN-AEL")

def get_loss(batch, vocab, encoder, decoder, loss_fn, args): 
    post_ids = [sentence2id(sent, vocab) for sent in batch[0]]
    # add GO
    response_ids = [[GO_ID] + sentence2id(sent, vocab) for sent in batch[1]]
    reference_ids = [sentence2id(sent, vocab) for sent in batch[1]]


    posts_var, posts_length = padding_inputs(post_ids, None)
    responses_var, responses_length = padding_inputs(response_ids, args.dec_max_len)
    # add EOS
    references_var, references_length = padding_inputs(reference_ids, args.dec_max_len, eos=True)

    #for q, r, t in zip(posts_var.data.numpy(), responses_var.data.numpy(), references_var.data.numpy()):
    #    print "".join(id2sentence(q, rev_vocab)).encode('utf-8')
    #    print "".join(id2sentence(r, rev_vocab)).encode('utf-8')
    #    print "".join(id2sentence(t, rev_vocab)).encode('utf-8')
    #    print '*' * 30

    # sort by post length
    posts_length, perms_idx = posts_length.sort(0, descending=True)
    posts_var = posts_var[perms_idx]
    responses_var = responses_var[perms_idx]
    responses_length = responses_length[perms_idx]
    references_var = references_var[perms_idx]
    references_length = references_length[perms_idx]

    if args.use_cuda:
        posts_var = posts_var.cuda()
        responses_var = responses_var.cuda()
        references_var = references_var.cuda()

    _, dec_init_state = encoder(posts_var, inputs_length=posts_length.numpy())
    output, state = decoder(dec_init_state, responses_var) # [B, T, vocab_size]
    
    loss = loss_fn(output.view(-1, output.size(2)),
                     references_var.view(-1))
    return loss

def early_stopping(eval_losses, diff):
    assert isinstance(diff, float), "early_stopping should be either None or float value. Got {}".format(diff)
    eval_loss_diff = np.abs(eval_losses[-2] - eval_losses[-1])
    if eval_loss_diff < diff:
        logger.info("Evaluation loss stopped decreased less than {}. Early stopping now.".format(diff))
        return True
    else:
        return False


def save_model(save_dir, epoch, 
               encoder, decoder, discriminator=None):
    torch.save(encoder.state_dict(), os.path.join(save_dir, 'epoch%d.encoder.params.pkl' % epoch))
    torch.save(decoder.state_dict(), os.path.join(save_dir, 'epoch%d.decoder.params.pkl' % epoch))
    if discriminator:
        torch.save(discriminator.state_dict(), os.path.join(save_dir, 'epoch%d.discriminator.params.pkl' % epoch))
    logger.info('Save model (epoch = %d) in %s' % (epoch, save_dir))


def reload_model(reload_dir, epoch, encoder, decoder, discriminator=None):
    try:
        encoder.load_state_dict(torch.load(
            os.path.join(reload_dir, 'epoch%d.encoder.params.pkl' % epoch)))
        decoder.load_state_dict(torch.load(
            os.path.join(reload_dir, 'epoch%d.decoder.params.pkl' % epoch)))
        if discriminator:
            discriminator.load_state_dict(torch.load(
                os.path.join(reload_dir, 'epoch%d.discriminator.params.pkl' % epoch)))
        logger.info("Loading parameters from %s in epoch %d" % (reload_dir, epoch))
    except:
        logger.info("No stored model to load from %s in epoch %d" % (reload_dir, epoch))
        print "reload error"
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
        loss = get_loss(batch, vocab, encoder, decoder, loss_fn, args)
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
                      args.embedding_dim, 2*args.hidden_dim, args.projection_dim, 
                      encoder.n_layers, args.dropout_p, args.rnn_cell, use_attention=False)
    
    if args.use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    
    return encoder, decoder

def build_gan():
    pass

def common_opt(parser):
    parser.add_argument('--data_name', help='the name of dataset such as weibo',
                        type=str, required=True)
    parser.add_argument('--mode', '-m', choices=('pretrain', 'adversarial', 'inference'),
                        type=str, required=True)
    parser.add_argument('--exp_dir', type=str, default=None)
    
    # train
    parser.add_argument('--batch_size', '-b', type=int, default=168)
    parser.add_argument('--num_epoch', '-e', type=int, default=10)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--use_cuda', default=True)
    parser.add_argument('--early_stopping', type=float, default=0.1)

    # resume
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_dir', type=str)
    parser.add_argument('--resume_epoch', type=int)

    # model
    parser.add_argument('--vocab_size', '-vs', type=int, default=100000)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--projection_dim', type=int, default=100)
    parser.add_argument('--dropout_p', '-dp', type=float, default=0.5)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--rnn_cell', type=str, default='gru')
    parser.add_argument('--dec_max_len', type=int, default=15)

def predict_opt(parser):
    parser.add_argument('--test_query_file', type=str, required=True)
    parser.add_argument('--load_path', type=str, required=True)
    # TODO: load epoch -> load best model
    parser.add_argument('--load_epoch', type=int, required=True)
    
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
    parser.add_argument('--training_ratio', type=int, default=2)
    parser.add_argument('--g_learning_rate', '-glr', type=float, default=0.001)
    parser.add_argument('--d_learning_rate', '-dlr', type=float, default=0.001)
    parser.add_argument('--filter_num', type=int, required=True)
    parser.add_argument('--filter_sizes', type=str, required=True)

     

class ApproximateEmbeddingLayer(nn.Module):
    r"""
    接收lstm的输出h_i, (batch_size, hid_size) * (hid_size, vocab_size)的Wp权重矩阵，-> (batch_size, vocab_size)
    经过归一化：softmax( (h_i + z_i)*W_p + b_p)之后，得到(batch_size, vocab_size) * (vocab_size, emb_size)再和word embeddings相乘，
    得到(batch_size, emb_size)即为当前时刻得到的approximate embeddings
    """
    def __init__(self, embedding, hidden_dim, vocab_size):
        super(ApproximateEmbeddingLayer, self).__init__()
        self.embedding = embedding
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.out = nn.Linear(hidden_dim, vocab_size)


    def forward(self, inputs):
        """
        Args:
            - **inputs** (batch_size, 1, hidden_dim)
        """
        noise = Variable(torch.rand(inputs.size()).normal_(0., 0.1), requires_grad=False)
        if inputs.is_cuda:
            noise = noise.cuda()
        score = self.out(inputs+noise)
        log_p = F.log_softmax(score)
        approximate_embeddings = torch.mm(F.softmax(score), self.embedding.weight) # 得到(batch_size, emb_size)
        return (normalized_weights, approximate_embeddings)


    def __repr__(self):
        s = '{name}({hidden_dim}, {vocab_size}'
        if self.hidden_dim is not None:
            s += ', hidden_dim={hidden_dim}'
        if self.vocab_size is not None:
            s += ', vocab_size={vocab_size}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

