# coding: utf-8

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from baseRNN import BaseRNN
from utils import ApproximateEmbeddingLayer


from utils import SYM_GO

class Generator(BaseRNN):
    """Generator """
    # baseRNN中SYM_PAD = 0

    def __init__(self, vocab_size, max_len, emb_dim, hidden_dim,
                n_layers=1, rnn_cell='gru', dropout_p=0, use_attention=False):
        super(Generator, self).__init__(vocab_size, hidden_dim, dropout_p, n_layers, rnn_cell)
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        # decoder一般都是单向的
        self.rnn = self.rnn_cell(emb_dim, hidden_dim, n_layers,
                                 batch_first=True, dropout=dropout_p, bidirectional=False)
        # approximate embedding layer
        self.ael = ApproximateEmbeddingLayer(hidden_dim, vocab_size)

    def forward(self, init_state, word_embeddings):
        """
        Args:
            init_state: 如果encoder的rnn cell选用lstm，则(ht, ct)；如果用gru，则ht
            shape都是(num_layers, batch, num_directions*hidden_size)
            NOTED: 最好是encoder选用几层，decoder就选择几层；decoder设置其hidden_size=encoder hidden_size*2
        传入word_embeddings矩阵后，首先得到go的输入vector，不断的rnn_cell生成出新词，decode到最大长度为止
        mode: 有三种模式，
        (1) 对抗训练: 这时decoder只需要y_input = [go]即可, 但reference = y
        """
        if isinstance(init_state, tuple):
            assert init_state[0].size(2) == self.hidden_dim, "If encoder is bidirectional, \
                decoder hidden size should be 2 * encoder hidden state."
            batch_size = init_state[0].size(1)
        else:
            assert init_state.size(2) == self.hidden_dim, "If encoder is bidirectional, \
                decoder hidden size should be 2 * encoder hidden state."
            batch_size = init_state.size(1)
        go_inputs = Variable(torch.ones(batch_size, 1).long()*SYM_GO, requires_grad=False).cuda()
        embedded_input = word_embeddings(go_inputs) # [B, 1, emb_dim]
        state = init_state
        outputs = [] # a list of approximate word embeddings(max_len)
        for i in range(self.max_len):
            output, state = self.rnn(embedded_input, state)
            # output = [B, 1, hidden_dim], state有可能是tuple, 每个元素(num_layers, B, hidden_dim)
            _, embedded_input = self.ael(output.squeeze(1), word_embeddings) # [B, emb_dim]
            outputs.append(embedded_input)
            embedded_input = embedded_input.unsqueeze(1) # [B, 1, emb_dim]
        return torch.stack(outputs, dim=1) # [B, max_len, emb_dim]
        # 输出的是每一时刻得到的approximate embedding

    def inference(self, init_state, word_embeddings):
        """
        Args:
            init_state: 如果encoder的rnn cell选用lstm，则(ht, ct)；如果用gru，则ht
            shape都是(num_layers, batch, num_directions*hidden_size)
            NOTED: 最好是encoder选用几层，decoder就选择几层；decoder设置其hidden_size=encoder hidden_size*2
        传入word_embeddings矩阵后，首先得到go的输入vector，不断的rnn_cell生成出新词，decode到最大长度为止
        mode: 有三种模式，
        (1) 对抗训练: 这时decoder只需要y_input = [go]即可, 但reference = y
        (2) 有监督预训练: 这时y_input=[go] + y; reference = y + [eos]
        (3) 预测: 这时只需要给[go], 不需要reference
        """
        if isinstance(init_state, tuple):
            assert init_state[0].size(2) == self.hidden_dim, "If encoder is bidirectional, \
                decoder hidden size should be 2 * encoder hidden state."
            batch_size = init_state[0].size(1)
        else:
            assert init_state.size(2) == self.hidden_dim, "If encoder is bidirectional, \
                decoder hidden size should be 2 * encoder hidden state."
            batch_size = init_state.size(1)
        go_inputs = Variable(torch.ones(batch_size, 1).long()*SYM_GO, requires_grad=False).cuda()
        embedded_input = word_embeddings(go_inputs) # [B, 1, emb_dim]
        state = init_state
        outputs = [] # a list of approximate word embeddings(max_len)
        for i in range(self.max_len):
            output, state = self.rnn(embedded_input, state)
            # output = [B, 1, hidden_dim], state有可能是tuple, 每个元素(num_layers, B, hidden_dim)
            word_distribution, embedded_input = self.ael(output.squeeze(1), word_embeddings) # [B, emb_dim]
            # [B, vocab_size]
            outputs.append(word_distribution)
            embedded_input = embedded_input.unsqueeze(1)
        return torch.stack(outputs, dim=1) # [B, max_len, vocab_size]
        # 预测阶段，输出的是每一时刻最大概率出现的词

    def update(self, s, embedded_i):
        # s = (output, st)
        # embedded_i = [B, T, emb_dim]
        return self.rnn(embedded_i, s[1])

    def predict(self, s, word_embeddings):
        logp, _ = self.ael(s[0].squeeze(1), word_embeddings)    
        return logp # [B, vocab_size]

    def supervise(self, dec_inputs, init_state, word_embeddings):
        """
        Args:
            init_state: 如果encoder的rnn cell选用lstm，则(ht, ct)；如果用gru，则ht
            shape都是(num_layers, batch, num_directions*hidden_size)
            NOTED: 最好是encoder选用几层，decoder就选择几层；decoder设置其hidden_size=encoder hidden_size*2
        传入word_embeddings矩阵后，首先得到go的输入vector，不断的rnn_cell生成出新词，decode到最大长度为止
        references = [B, T, emb_dim]

        mode: 有三种模式，
        (1) 对抗训练: 这时decoder只需要y_input = [go]即可, 但reference = y
        (2) 有监督预训练: 这时y_input=[go] + y; reference = y + [eos]
        (3) 预测: 这时只需要给[go], 不需要reference
        """
        # assert references.size(1) == (self.max_len - 1), "When supervise learning, length of references should be (max_len - 1)."
        if isinstance(init_state, tuple):
            assert init_state[0].size(2) == self.hidden_dim, "If encoder is bidirectional, \
                decoder hidden size should be 2 * encoder hidden state."
            batch_size = init_state[0].size(1)
        else:
            assert init_state.size(2) == self.hidden_dim, "If encoder is bidirectional, \
                decoder hidden size should be 2 * encoder hidden state."
            batch_size = init_state.size(1)
        ref_inputs = list(torch.split(dec_inputs, 1, dim=1)) # a list of [B, 1, emb_dim]
        go_inputs = Variable(torch.ones(batch_size, 1).long()*SYM_GO, requires_grad=False).cuda()
        embedded_inputs = [word_embeddings(go_inputs)] + ref_inputs
        state = init_state
        outputs = [] # a list of approximate word embeddings(max_len)
        for i in range(self.max_len+1):
            output, state = self.rnn(embedded_inputs[i], state)
            # output = [B, 1, hidden_dim], state有可能是tuple, 每个元素(num_layers, B, hidden_dim)
            word_distribution, _ = self.ael(output.squeeze(1), word_embeddings) # [B, emb_dim]
            # [B, vocab_size]
            outputs.append(word_distribution)
            # embedded_input = embedded_input.unsqueeze(1)
        return torch.stack(outputs, dim=1) # [B, max_len+1, vocab_size]
        # 有监督阶段，输出是词向量分布概率，后续需要计算NLL

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)


if __name__ == '__main__':
    vocab_size = 10
    max_len = 8
    emb_dim = 6
    hidden_dim = 5*2
    n_layers = 3
    # batch = 4
    decoder = Generator(vocab_size, max_len, emb_dim, hidden_dim, n_layers, rnn_cell='gru')
    init_state_ht = Variable(torch.rand(n_layers, 4, hidden_dim).uniform_(-0.05, 0.05))
    # init_state_ct = Variable(torch.rand(3, 4, 5*2).uniform_(-0.05, 0.05))
    embeddings = nn.Embedding(vocab_size, emb_dim)
    # embedded_inputs = embeddings(inputs)
    res = decoder(init_state_ht, embeddings)
    print res.shape

