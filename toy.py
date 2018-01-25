# coding:utf-8

import sys
import math

from discriminator import Discriminator
from generator import Generator
from encoder import EncoderRNN, padding_inputs

import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import SYM_PAD, SYM_GO, SYM_EOS

from data import batcher, build_vocab, load_vocab, sentence2id

# word_embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx=SYM_PAD)
# E = EncoderRNN(vocab_size, emb_dim, hidden_dim, n_layers, bidirectional=True, variable_lengths=True)
# G = Generator(vocab_size, response_max_len, emb_dim, 2*hidden_dim, n_layers)
# D = Discriminator(emb_dim, filter_num=30, filter_sizes=[1,2,3,4])


def mask(x):
    """
    返回x的mask矩阵，即x中为0的部分，全部为0，非0的部分，全部为1
    """
    return x.gt(Variable(torch.zeros(x.size()).long())).float()

def eval():
    pass

def pretrain():
    # post_sentences, response_sentences = load_data_from_file('toy_data')
    batch_size = 32
    num_epoch = 20
    query_file = 'dataset/post.test'#stc_weibo_train_post'
    response_file = 'dataset/response.test'#stc_weibo_train_response'
    vocab_file = 'vocab.172'
    

    if not vocab_file:
        print "no vocabulary file"
        build_vocab(query_file, response_file, seperated=True)
        sys.exit()
    else:
        vocab, rev_vocab = load_vocab(vocab_file, max_vocab=100000)

    # vocab, rev_vocab = build_vocab(post_sentences, response_sentences)

    # return post_ids, response_ids # 二维列表(int)

    vocab_size = len(vocab)
    # 这里的max_len仅仅影响decoder的最大长度
    response_max_len = 20
    emb_dim = 128
    hidden_dim = 512
    n_layers = 2

    # a, l, b = sort_inputs(post_ids, response_ids, response_max_len)
    # 有三种模式：对抗训练(adversarial)，预测(predict)，有监督训练(supervise)
    

    # 此时mode有两种: supervise, adversarial
    # assert mode, 'when training, please assign mode (adversarial/supervise)'
    LR_G = 0.0001  # learning rate for generator
    # LR_D = 0.0001  # learning rate for discriminator
    # opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
    # opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)
    word_embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx=SYM_PAD)
    E = EncoderRNN(vocab_size, emb_dim, hidden_dim, n_layers, bidirectional=True, variable_lengths=True)
    G = Generator(vocab_size, response_max_len, emb_dim, 2*hidden_dim, n_layers)

    

    loss_func = nn.NLLLoss()
    params = list(word_embeddings.parameters()) + list(E.parameters()) + list(G.parameters())
    opt = torch.optim.Adam(params, lr=LR_G)

    for e in range(num_epoch):
        data_generator = batcher(query_file, response_file, batch_size)
        # 1. 从dataset中generate一个batch数据
        # 2. 生成fake response
        # 3. 分别用判别器判断prob_0和prob_1
        # 这里的sort_inputs在padding时，按照当前batch中句子最大长度pad的
        print "Epoch %d:" % e
        step = 0
        while True:
            try:
                post_sentences, response_sentences = data_generator.next()
            except StopIteration:
                # one epoch finish
                eval()
                break


            post_ids = [sentence2id(sent, vocab) for sent in post_sentences]
            response_ids = [sentence2id(sent, vocab) for sent in response_sentences]
            posts_var, posts_length, responses_var, responses_length = padding_inputs(post_ids, response_ids, response_max_len)
            # print "posts_var shape:", posts_var.size()
            # print "responses_var shape:", responses_var.size()
            # 在sentence后面加eos
            references_var = torch.cat([responses_var, Variable(torch.zeros(responses_var.size(0),1).long())], dim=1)
            for idx, length in enumerate(responses_length):
                references_var[idx, length] = SYM_EOS

            embedded_post = word_embeddings(posts_var)
            embedded_response = word_embeddings(responses_var)
            # embedded_ref = word_embeddings(references_var)

            _, dec_init_state = E(embedded_post, input_lengths=posts_length.numpy())
            # G_ideas = Variable(torch.randn(BATCH_SIZE, N_IDEAS))    # random ideas
            output_dist = G.supervise(embedded_response, dec_init_state, word_embeddings) # [B, T, vocab_size]
            # print output_dist
            mask_pos = mask(references_var).unsqueeze(-1).expand_as(output_dist)
            # print mask_pos

            # output_dist = [B, T, vocab_size]
            # mask = [B, T]
            masked_output = output_dist*mask_pos

            # [B*T, vocab_size]
            loss = loss_func(masked_output.view(-1, vocab_size), references_var.view(-1))/posts_var.size(0)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # if step % 50 == 0:
            char_level_loss = loss/(mask_pos.mean())
            print 'Step %d, Perplexity: %.2f' % (step, math.exp(char_level_loss))
            step = step + 1

if __name__ == '__main__':
    pretrain()


"""
对抗
prob_real = D(embedded_post, real_responses)          # D try to increase this prob
prob_fake = D(embedded_post, fake_responses)               # D try to reduce this prob



opt_D.zero_grad()
D_loss.backward(retain_variables=True)      # retain_variables 这个参数是为了再次使用计算图纸
opt_D.step()

opt_G.zero_grad()
G_loss.backward()
opt_G.step()

if step % 50 == 0:  # plotting
    print 'D score = %.2f' % -D_loss.data.numpy()
    print 'D accuracy = %.2f' % prob_real.data.numpy().mean()
    for sent in G.inference(dec_init_state, word_embeddings).data.numpy():
        print ' '.join(id2sentence(sent.tolist(), rev_vocab))
"""
