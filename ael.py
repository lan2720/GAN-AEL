import torch
import torch.nn as nn
import torch.nn.functional as F


class ApproximateEmbeddingLayer(nn.Module):
    def __init__(self, embedding):
        super(ApproximateEmbeddingLayer, self).__init__()
        self.embedding = embedding


    def forward(self, inputs):
        """
        Args:
            - **inputs** (batch_size, 1, vocab_size)
        """
        #noise = Variable(torch.rand(inputs.size()).normal_(0., 0.1), requires_grad=False)
        #if inputs.is_cuda:
        #    noise = noise.cuda()
        #score = self.out(inputs+noise)
        assert inputs.size(1) == 1
        inputs = inputs.squeeze(1)
        log_p = F.log_softmax(inputs) # (B, V)
        approximate_embeddings = torch.mm(F.softmax(inputs), self.embedding.weight) # (batch_size, emb_size)
        return approximate_embeddings.unsqueeze(1) # [b, 1, emb_dim]

