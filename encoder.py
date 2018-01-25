# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable

from baseRNN import BaseRNN

class EncoderRNN(BaseRNN):
    def __init__(self, vocab_size, emb_dim, hidden_dim,
            n_layers=1, dropout_p=0, bidirectional=False, rnn_cell='gru', variable_lengths=False):
        super(EncoderRNN, self).__init__(vocab_size, hidden_dim, dropout_p, n_layers, rnn_cell)

        self.variable_lengths = variable_lengths
        self.bidirectional = bidirectional
        # embedding共享，因此从外部输入
        # self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=self.SYM_PAD)
        self.rnn = self.rnn_cell(emb_dim, hidden_dim, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

    def forward(self, embedded_inputs, input_lengths=None):
        # input_var都是经过pad的，只不过在变长时，会输入input真实长度input_lengths，如果input_lengths=None即为等长
        # embedded = word_embeddings(input_var)
        # embedded_inputs = [B, T, D]
        if self.variable_lengths:
            assert input_lengths is not None, "when input's length is variable, 'input_lengths' is needed when using EncoderRNN."
            embedded_inputs = nn.utils.rnn.pack_padded_sequence(embedded_inputs, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded_inputs)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # 输出shape=(batch, seq_len, hidden_size * num_directions)，仅供decoder中attention使用
        # 如果是变长，则输出中存在很多全0部分

        # 如果是双向，最后一个时刻，hidden(ht, 或ht和ct) = (batch, num_layers * num_directions, hidden_size)
        #             需要变成(batch, num_layers, hidden_size * num_directions)
        if isinstance(hidden, tuple):
            hidden = tuple(map(self._fix_hidden, hidden))
        else:
            hidden = self._fix_hidden(hidden)
        # 这其中，由于batch中各个case长度不同，hidden中存在有些行是全0的，那么，给decoder初始化时也都是全0初始化
        return output, hidden


    def _fix_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        # 方便decoder的初始化
        if self.bidirectional:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h


def padding_inputs(x, y, y_max_len):
    """
    x,y: 均为整型的二维矩阵
    y_max_len: 由于x根据一个batch中最长的句子进行padding，因此不需要给定max_len，但是y需要给定
    """
    # x 整型二维列表
    x_lens = torch.LongTensor(map(len, x))
    y_lens = torch.zeros(len(y)).long()
    x_inputs = Variable(torch.zeros(len(x), max(x_lens)).long(), requires_grad=False)
    y_inputs = Variable(torch.zeros(len(y), y_max_len).long(), requires_grad=False)

    # 输入word id本身已经用0 padding
    for idx, (seq, seq_len) in enumerate(zip(x, x_lens)):
        x_inputs[idx, :seq_len] = torch.LongTensor(seq)

    for idx, seq in enumerate(y):
        if len(seq) >= y_max_len:
            y_inputs[idx] =  torch.LongTensor(seq[:y_max_len])
            y_lens[idx] = y_max_len
        else:
            y_inputs[idx, :len(seq)] = torch.LongTensor(seq)
            y_lens[idx] = len(seq)


    x_lens, perms_idx = x_lens.sort(0, descending=True)
    x_inputs = x_inputs[perms_idx]
    y_inputs = y_inputs[perms_idx]
    y_lens = y_lens[perms_idx]

    return x_inputs, x_lens, y_inputs, y_lens


if __name__ == '__main__':
    x_seqs = [[5,3,2,4], [3,4,8,1,5,3,1], [4,3]]
    y_seqs = [[1,2,3], [5,4,7,4,6,8,6], [2,3,6,7,3]]
    a, a_len, b, b_len = padding_inputs(x_seqs, y_seqs, 10)
    print a
    print a_len
    print b
    print b_len
    # seq_lens = torch.LongTensor(map(len, seqs))
    # inputs = Variable(torch.zeros(len(seqs), max(seq_lens))).long()
    # # 输入word id本身已经用0 padding
    # for idx, (seq, seq_len) in enumerate(zip(seqs, seq_lens)):
    #     inputs[idx, :seq_len] = torch.LongTensor(seq)

    # seq_lens, perms_idx = seq_lens.sort(0, descending=True)
    # inputs = inputs[perms_idx]


    # vocab_size = 10
    # emb_dim = 6
    # hidden_dim = 5
    # embeddings = nn.Embedding(vocab_size, emb_dim)
    # embedded_inputs = embeddings(inputs)
    # print "embedded input shape:", embedded_inputs.shape
    # print "embedded input", embedded_inputs
    # print "*"*20

    # encoder = EncoderRNN(vocab_size, emb_dim, hidden_dim, bidirectional=True, variable_lengths=True)
    # output, hidden = encoder(embeddings, inputs, input_lengths=seq_lens.numpy())
    # # 如果给定了input_lengths, 则encoder自动会在长度以外的部分输出全0
    # # 如果不给input_lengths, 则encoder会用id=0的word vector对padding部分也同样进行rnn运算
    # print hidden

    # lstm = nn.LSTM(emb_dim, hidden_dim, 4, batch_first=True, bidirectional=True)
    # output, (ht, ct) = lstm(embedded_inputs)
    

    # print "lstm result shape:", output.shape
    # print output
    # print "*"*20
    # print "lstm result shape:", ht.shape
    # # print "lstm result shape:", ct.shape
    # print ht
    # print "*"*20

    # ht = func(ht)
    # print "after translate:", ht.shape
    # print ht


