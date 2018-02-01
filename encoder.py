# coding:utf-8

import utils

import torch
import torch.nn as nn
from baseRNN import BaseRNN


class EncoderRNN(BaseRNN):
    def __init__(self, 
                vocab_size,
                emb_dim,
                hidden_dim,
                n_layers=1,
                dropout_p=0,
                rnn_cell='gru'):
        super(EncoderRNN, self).__init__(vocab_size, emb_dim, hidden_dim, n_layers, dropout_p, rnn_cell)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=utils.PAD_ID)
        self.rnn = self.rnn_cell(emb_dim, hidden_dim, n_layers,
                                 batch_first=True, dropout=dropout_p, bidirectional=True)

    def forward(self, inputs, inputs_length):
        """
        Args:
            - **inputs** (batch_size, max_len)
            - **inputs_length** (batch_size,)
        Outputs:
           - **output** only useful when attention 
           - **hidden** (n_layers, batch_size, (2*hidden_dim))
        """
        embedded_inputs = self.embedding(inputs)
        embedded_inputs = nn.utils.rnn.pack_padded_sequence(embedded_inputs, inputs_length, batch_first=True)
        output, state = self.rnn(embedded_inputs)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        if isinstance(state, tuple):
            state = tuple(map(self._fix_hidden, state))
        else:
            state = self._fix_hidden(state)
        return output, state
    

    def _fix_hidden(self, h):
        #[batch, (layers*directions), dim] -> [layers, batch, (directions*dim)]
        return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)

