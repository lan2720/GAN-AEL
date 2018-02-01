# coding:utf-8

import jieba
from collections import Counter
import itertools

import torch
from torch.autograd import Variable

def batcher(batch_size, query_file, response_file=None, seperated=True):
    queries = []
    fq = open(query_file, 'rb')
    
    if response_file:
        responses = []
        fr = open(response_file, 'rb')
    
    while True:
        qline = fq.readline()
        if qline == '':
            break

        if seperated:
            queries.append(qline.strip().decode('utf-8').split())
        else:
            queries.append(list(jieba.cut(qline.strip()), cut_all=False))
        
        if response_file:
            rline = fr.readline()
            if seperated:
                responses.append(rline.strip().decode('utf-8').split())
            else:
                responses.append(list(jieba.cut(rline.strip()), cut_all=False))

        if len(queries) == batch_size:
            if response_file:
                assert len(queries) == len(responses), 'the size of queries and \
                    the size of responses should be the same in one batch.'
                yield (queries, responses)
                queries = []
                responses = []
            else:
                yield queries
                queries = []

    fq.close()
    if response_file:
        fr.close()
    if queries:
        if response_file:
            assert len(queries) == len(responses), 'the size of queries and \
                the size of responses should be the same in one batch.'
            yield (queries, responses)
        else:
            yield queries
           

def pretty_print(queries, responses):
    data = []
    for q, r in zip(queries, responses):
        data.append(''.join(q) + ' -> ' + ''.join(r))
    print '\n'.join(data)

def read_data(file, seperated=True):
    data = [] # 字符串类型的二维列表
    with open(file, 'rb') as f:
        for line in f:
            if seperated:
                data.append(line.strip().decode('utf-8').split())
            else:
                data.append(list(jieba.cut(line.strip()), cut_all=False))
    return data


def load_data_from_file(filename):
    posts = []
    responses = []
    with open(filename, 'rb') as f:
        for line in f:
            parts = line.strip().split('\t')
            posts.append(parts[0].decode('utf-8').split())
            responses.append(parts[1].decode('utf-8').split())
    # posts是个二维list,每个list中都是分好的词
    return (posts, responses)

def split_words(sentence):
    seg_list = jieba.cut(sentence, cut_all=False)
    return seg_list

def build_vocab(query_file, response_file, seperated=True):
    print('Build vocabulary from:\nq: %s\nr: %s' % (query_file, response_file))
    queries = read_data(query_file, seperated)
    responses = read_data(response_file, seperated)
    vocab = Counter(itertools.chain.from_iterable(queries+responses))
    vocab = [w for w, c in vocab.most_common()]

    vocab = ['<PAD>', '<GO>', '<EOS>', '<UNK>'] + vocab # pad id = 0
    with open('vocab.%d' % len(vocab), 'wb') as fo:
        fo.write('\n'.join(vocab).encode('utf-8'))
    word2idx = dict([(w, i) for i, w in enumerate(vocab)])
    idx2word = dict([(i, w) for i, w in enumerate(vocab)])
    print('Vocab size: %d' % len(vocab))
    print('Dump to vocab.%d' % len(vocab))
    return word2idx, idx2word

def load_vocab(vocab_file, max_vocab=100000):
    words = list(itertools.chain.from_iterable(read_data(vocab_file)))
    if len(words) > max_vocab:
        words = words[:max_vocab]
    vocab = dict([(w, i) for i, w in enumerate(words)])
    rev_vocab = dict([(i, w) for i, w in enumerate(words)])
    return vocab, rev_vocab

def padding_inputs(x, max_len=None):
    """
    x: 均为整型的二维矩阵
    max_len: default=None 由于x根据一个batch中最长的句子进行padding，因此不需要给定max_len
    when max_len != None, padding according to the max_len
    """
    # x 整型二维列表
    x_lens = torch.LongTensor(map(len, x))
    if not max_len:
        max_len = max(x_lens)
    
    x_inputs = Variable(torch.zeros(len(x), max_len).long(), requires_grad=False)

    # 输入word id本身已经用0 padding
    for idx, (seq, seq_len) in enumerate(zip(x, x_lens)):
        if seq_len > max_len:
            x_inputs[idx, :max_len] = torch.LongTensor(seq[:max_len])
            x_lens[idx] = max_len
        else:
            x_inputs[idx, :seq_len] = torch.LongTensor(seq)

    return x_inputs, x_lens

def sentence2id(words, vocab):
    return [vocab.get(w, vocab.get('<UNK>')) for w in words]

def id2sentence(ids, rev_vocab):
    return [rev_vocab.get(i) for i in ids]

if __name__ == '__main__':
    b = batcher('dataset/post.test', 'dataset/response.test', batch_size=3, num_epoch=2)


