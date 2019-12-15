from torch.utils.data import Dataset
import torch

from .preprocessing import WordEmbedding, load_word_emb

w2v_config = {
    'data_dir': 'data/glove',
    'word2idx_path': 'word2idx.json',
    'usedwordemb_path': 'usedwordemb.npy'
}

class AlphabetSortingDataset(Dataset):
    
    def __init__(self, num_samples, min_len=20, max_len=60, alphabet='abcdefghijklmnopqrstuvwxyz'):
        w2v = WordEmbedding(load_word_emb(w2v_config['data_dir'], 
                                          w2v_config['word2idx_path'],
                                          w2v_config['usedwordemb_path'])
                           )
        
        self.num_samples = num_samples
        self.alphabet = alphabet
        self.embedding = {char: w2v(char) for char in self.alphabet}
        self.min_len = min_len
        self.max_len = max_len
        self.x, self.chars, self.y = self._build()
        
    def _build(self):
        array_len = torch.randint(low=self.min_len, 
                            high=self.max_len + 1,
                            size=(1,))
        x_idxs = torch.randint(high=len(self.alphabet), size=(self.num_samples, array_len))
        y = x_idxs.argsort()
        x, chars = [], []
        for idxs in x_idxs:
            x_appendix, chars_appendix = [], []
            for idx in idxs:
                char = self.alphabet[idx]
                chars_appendix.append(char)
                x_appendix.append(self.embedding[char])
            x.append(x_appendix)
            chars.append(chars_appendix)
        return torch.Tensor(x), chars, y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.chars[idx]
    
class NumberSortingDataset(Dataset):
    
    def __init__(self, num_samples, min_num=0, max_num=9, min_len=20, max_len=60):
        self.num_samples = num_samples
        self.min = min_num
        self.max = max_num
        self.min_len = min_len
        self.max_len = max_len
        self.x, self.y = self._build()
        
    def _build(self):
        array_len = torch.randint(low=self.min_len, 
                            high=self.max_len + 1,
                            size=(1,))
        x = torch.randint(low=self.min, high=self.max, size=(self.num_samples, array_len))
        y = x.argsort()
        return x, y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]