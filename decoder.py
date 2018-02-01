# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from baseRNN import BaseRNN


class DecoderRNN(BaseRNN):
    def __init__(self,
                max_len,
                embedding,
                vocab_size,
                emb_dim,
                hidden_dim,
                proj_dim,
                n_layers=1, 
                dropout_p=0,
                rnn_cell='gru',
                use_attention=False):
        super(DecoderRNN, self).__init__(vocab_size, emb_dim, hidden_dim, n_layers, dropout_p, rnn_cell)
        self.max_len = max_len
        self.embedding = embedding
        self.rnn = self.rnn_cell(emb_dim, hidden_dim, n_layers,
                                 batch_first=True, dropout=dropout_p, bidirectional=False)
        self.dropout = nn.Dropout(dropout_p)
        self.proj = nn.Linear(hidden_dim, proj_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        #self.ael = ApproximateEmbeddingLayer(hidden_dim, vocab_size)

    def forward(self, state, inputs):
        """
        Args:
            - **init_state** ht or (ht, ct) (n_layers, batch, num_directions*hidden_size)
            - **inputs** (batch_size, 1/max_len) ( GO, id1, id2, ... )
        Outputs:
            - ****:
            - ****:
        """
        embedded_inputs = self.embedding(inputs) # [B, 1/max_len, emb_dim]
        output, state = self.rnn(embedded_inputs, state)
        return self.out(output), state
    
    def update(self, s, xi):
        # s = (output, st)
        # xi = [B, 1]
        """
        Args:
            - **s** Tuple (output, st)
            - **xi** [B, 1]
        Outputs:
            [B, 1, hidden_dim]
        """
        embedded_i = self.embedding(xi)
        return self.rnn(embedded_i, s[1])


    def predict(self, s):
        """
        Args:
            - **state**  tuple (output, state)
        Outputs:
            [B, 1, vocab_size]
        """
        output = self.out(s[0]).squeeze(1) # [B, vocab_size]
        log_p = F.log_softmax(output, dim=1)
        return log_p

