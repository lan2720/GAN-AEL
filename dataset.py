import linecache
from torch.utils.data import Dataset, DataLoader

def tokenizer(line):
    return line.strip().split()

#class DialogDataset(Dataset):
#    def __init__(self, query_path, resp_path, process_fn=None):
#        self.query_path = query_path
#        self.resp_path = resp_path
#        self.process_fn = process_fn
#        self.total_data = 0
#
#        with open(query_path, 'rb') as fq, open(resp_path, 'rb') as fr:
#            q_size = len(fq.readlines())
#            r_size = len(fr.readlines())
#            assert q_size == r_size
#        self.total_data = q_size - 1
#
#    def __getitem__(self, index):
#        query = linecache.getline(self.query_path, index+1)#.encode('utf-8')
#        response = linecache.getline(self.resp_path, index+1)#.encode('utf-8')
#        
#        query = self.process_fn(query)
#        response = self.process_fn(response)
#        return (query, response)
#
#    def __len__(self):
#        return self.total_data

class DialogDataset(Dataset):
    def __init__(self, path, process_fn=None):
        self.path = path
        self.process_fn = process_fn
        self.total_data = 0

        with open(path, 'rb') as f:
            self.total_data = len(f.readlines()) - 1

    def __getitem__(self, index):
        line = linecache.getline(self.path, index+1)#.encode('utf-8')
        
        if self.process_fn:
            q, r = line.split('#')
            return (self.process_fn(q), self.process_fn(r))
        else:
            return line.strip().split('#')

    def __len__(self):
        return self.total_data

def test():
    #dset = DialogDataset('dataset/weibo/stc_weibo_train_100w.post', 'dataset/weibo/stc_weibo_train_100w.response', tokenizer)
    dset = DialogDataset('dataset/weibo/sample',  None)
    train_loader = DataLoader(dset, batch_size=3, shuffle=False, num_workers=4)

    for i, (q_b, r_b) in enumerate(train_loader, 1):
        print q_b
        print r_b
        break

if __name__ == '__main__':
    test()
