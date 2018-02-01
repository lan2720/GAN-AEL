# coding:utf-8

import utils
import torch.nn as nn


class GanAELModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(GanAELModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    

    def loss(self, x, x_len, y, t):
        # TODO
        loss_func = nn.CrossEntropyLoss(ignore_index=utils.PAD_ID) # word level
        ey, es = self.encoder(x, x_len)
        dy, ds = self.decoder(es, y)
        loss = loss_func(dy.view(-1, dy.size(-1)), t.view(-1)) 
        return es, ds, loss
    

