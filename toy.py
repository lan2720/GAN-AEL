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

from data import batcher, build_vocab, load_vocab, sentence2id, id2sentence

# word_embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx=SYM_PAD)
# E = EncoderRNN(vocab_size, emb_dim, hidden_dim, n_layers, bidirectional=True, variable_lengths=True)
# G = Generator(vocab_size, response_max_len, emb_dim, 2*hidden_dim, n_layers)
# D = Discriminator(emb_dim, filter_num=30, filter_sizes=[1,2,3,4])


def mask(x):
    """
    返回x的mask矩阵，即x中为0的部分，全部为0，非0的部分，全部为1
    """
    return x.gt(Variable(torch.zeros(x.size()).long()).cuda()).float()

def eval():
    pass

def pretrain():
    # post_sentences, response_sentences = load_data_from_file('toy_data')
    use_cuda = True
    batch_size = 128
    num_epoch = 20
    query_file = 'dataset/weibo/stc_weibo_train_post'
    response_file = 'dataset/weibo/stc_weibo_train_response'
    vocab_file = 'vocab.708320'#'vocab.172'
    

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
    response_max_len = 15
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
    
    if use_cuda:
        word_embeddings.cuda()
        E.cuda()
        G.cuda()
    

    loss_func = nn.CrossEntropyLoss(size_average=False)
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
            references_var = torch.cat([responses_var, Variable(torch.zeros(responses_var.size(0),1).long(), requires_grad=False)], dim=1)

            for idx, length in enumerate(responses_length):
                references_var[idx, length] = SYM_EOS

            #for p, r_in, r_out in zip(posts_var.data, responses_var.data, references_var.data):
            #    print 'q: ' + ''.join(id2sentence(p, rev_vocab))
            #    print 'r_in: ' + ''.join(id2sentence(r_in, rev_vocab))
            #    print 'r_out: ' + ''.join(id2sentence(r_out, rev_vocab))
            #    print '*'*30


            if use_cuda:
                posts_var = posts_var.cuda()
                #posts_length = posts_length.cuda()
                responses_var = responses_var.cuda()
                #responses_length = responses_length.cuda()
                references_var = references_var.cuda()

            embedded_post = word_embeddings(posts_var)
            embedded_response = word_embeddings(responses_var)
            # embedded_ref = word_embeddings(references_var)

            _, dec_init_state = E(embedded_post, input_lengths=posts_length.numpy())
            # G_ideas = Variable(torch.randn(BATCH_SIZE, N_IDEAS))    # random ideas
            outputs = G.supervise(embedded_response, dec_init_state, word_embeddings) # [B, T, vocab_size]
            # [B*T, vocab_size]
            # [B*T, vocab_size] mask 
            # [B*T, 1]
            outputs = outputs.view(-1, vocab_size)
            mask_pos = mask(references_var).view(-1).unsqueeze(-1)
            
            # print mask_pos

            # output_dist = [B, T, vocab_size]
            # mask = [B, T]
            masked_output = outputs*(mask_pos.expand_as(outputs))

            # [B*T, vocab_size]

            loss = loss_func(masked_output, references_var.view(-1))/(posts_var.size(0))

            opt.zero_grad()
            loss.backward()
            opt.step()

            # if step % 50 == 0:
            #char_level_loss = loss/(mask_pos.mean())
            #loss_val = char_level_loss.cpu().data.numpy()[0]
            ave_seq_len = torch.sum(mask_pos)/(posts_var.size(0))
            char_loss_val = loss/ave_seq_len
            #print type(char_level_loss)
            print 'Step %d, Loss: %.5f Perplexity: %.2f' % (step, loss.cpu().data.numpy()[0], math.exp(char_loss_val.cpu().data.numpy()[0]))
            step = step + 1

if __name__ == '__main__':
    pretrain()
    print "test"


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
