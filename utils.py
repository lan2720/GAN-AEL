# coding: utf-8

import torch

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import torch.nn as nn
from torch.nn.init import uniform, normal


SYM_PAD = 0
SYM_GO = 1
SYM_EOS = 2
SYM_UNK = 3

class ApproximateEmbeddingLayer(nn.Module):
    r"""
    接收lstm的输出h_i, (batch_size, hid_size) * (hid_size, vocab_size)的Wp权重矩阵，-> (batch_size, vocab_size)
    经过归一化：softmax( (h_i + z_i)*W_p + b_p)之后，得到(batch_size, vocab_size) * (vocab_size, emb_size)再和word embeddings相乘，
    得到(batch_size, emb_size)即为当前时刻得到的approximate embeddings
    """
    def __init__(self, hidden_dim, vocab_size):
        super(ApproximateEmbeddingLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.out = nn.Linear(hidden_dim, vocab_size)

    def reset_parameters(self):
        normal(self.weight.data)
        normal(self.noise.data)
        normal(self.bias.data)


    def forward(self, inputs, embeddings):
        # input shape = (batch_size, hid_size)
        #noise = Variable(torch.rand(inputs.size()).normal_(0., 0.1), requires_grad=False).cuda() # h_i = (batch_size, hid_dim)每行对应相加
        noise = Variable(torch.zeros(inputs.size()), requires_grad=False).cuda() # h_i = (batch_size, hid_dim)每行对应相加
        score = self.out(inputs+noise)
        normalized_weights = F.log_softmax(score)
        approximate_embeddings = torch.mm(F.softmax(score), embeddings.weight) # 得到(batch_size, emb_size)
        return (normalized_weights, approximate_embeddings)


    def __repr__(self):
        s = '{name}({hidden_dim}, {vocab_size}'
        if self.hidden_dim is not None:
            s += ', hidden_dim={hidden_dim}'
        if self.vocab_size is not None:
            s += ', vocab_size={vocab_size}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

