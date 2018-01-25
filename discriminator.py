# coding:utf-8
import torch
import torch.nn as nn
from textcnn import TextCNN

class Discriminator(nn.Module):
    """Discriminator """
    def __init__(self, emb_dim, filter_num, filter_sizes):
        super(Discriminator, self).__init__()
        self.query_cnn = TextCNN(emb_dim, filter_num, filter_sizes)
        self.response_cnn = TextCNN(emb_dim, filter_num, filter_sizes)
        self.judger = nn.Sequential(
                        nn.Linear(2*filter_num*len(filter_sizes), 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1),
                        nn.Sigmoid()
                    )

    def forward(self, query, response):
        # query is [B, max_len] (after padding)
        # embedded_query = word_embeddings(query)
        # embedded_response = word_embeddings(response)

        query_features = self.query_cnn(query) # [B, T, D] -> [B, all_features]
        response_features = self.response_cnn(response)

        inputs = torch.cat((query_features, response_features), 1)

        return self.judger(inputs)

if __name__ == '__main__':
    from torch.autograd import Variable
    from visualize import make_dot
    batch_size = 3
    max_len = 8
    vocab_size = 10
    emb_dim = 6
    query = Variable(torch.LongTensor([[1,2,4,5,3,0,0,0], [6,3,3,5,0,0,0,0], [8,6,0,0,0,0,0,0]]))
    # reference = Variable(torch.LongTensor([[6,3,3,6,8,3,0,0], [2,1,5,0,0,0,0,0], [9,1,7,4,2,6,9,2]]))
    fake = Variable(torch.LongTensor([[6,3,3,6,8,3,0,0], [2,1,5,0,0,0,0,0], [9,1,7,4,2,6,9,2]]))
    # fake = Variable(torch.LongTensor([[1,2,8,3,6,3,2,0], [2,1,2,2,4,6,7,9], [4,2,2,6,0,0,0,0]]))

    embeddings = nn.Embedding(vocab_size, emb_dim)

    d = Discriminator(emb_dim, filter_num=100, filter_sizes=[1,2,3,4])
    print d
    a = d(query, fake, embeddings)
    # make_dot(a).view()


