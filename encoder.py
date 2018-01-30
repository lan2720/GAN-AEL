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
        #hidden = self.init_hidden()
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
    
         
    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_dim))
        hidden = hidden.cuda()
        return hidden

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


