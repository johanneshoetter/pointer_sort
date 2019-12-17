from torch.utils.data import Dataset
import torch
#import pickle
from .preprocessing import WordEmbedding, load_word_emb
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

w2v_config = {
    'data_dir': 'data/glove',
    'word2idx_path': 'word2idx.json',
    'usedwordemb_path': 'usedwordemb.npy'
}

# new version: creates sequences with unique characters (can not create sequences longer than 26 chars)
class AlphabetSortingDataset(Dataset):
    
    #def __init__(self, num_samples, min_len=3, max_len=4, alphabet='0123456789'):
    def __init__(self, num_samples, min_len=3, max_len=4, alphabet='abcdefghijklmnopqrstuvwxyz'):
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
        def create_sequence(alphabet, length):
            return np.random.choice(alphabet, size=length, replace=False)
      
        array_len = np.random.randint(low=self.min_len, 
                            high=min(len(self.alphabet), self.max_len + 1))
        idx_pool =  list(range(0, len(self.alphabet)))
        x_idxs = torch.Tensor([create_sequence(idx_pool, array_len) for _ in range(self.num_samples)])
        y = x_idxs.argsort()
        x, chars = [], []
        for idxs in x_idxs:
            x_appendix, chars_appendix = [], []
            for idx in idxs:
                char = self.alphabet[int(idx.numpy())]
                chars_appendix.append(char)
                x_appendix.append(self.embedding[char])
            x.append(x_appendix)
            chars.append(chars_appendix)
        return torch.Tensor(x), chars, y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.chars[idx]
    
    def save(self, path):
        data = {
            'x': self.x,
            'y': self.y,
            'chars': self.chars,
            'min_len': self.min_len,
            'max_len': self.max_len
        }
        torch.save(data, os.path.join(path, 'alphabet.pt'))

    def load(self, path):
        self.x, self.y, self.chars = [], [], []
        data = torch.load(os.path.join(path, 'alphabet.pt'))
        x, y, chars = data['x'], data['y'], data['chars']
        self.x.extend(x)
        self.y.extend(y)
        self.chars.extend(chars) 
        return data['min_len'], data['max_len']
    
    def split(self, train_ratio):
        train_idx, test_idx = train_test_split([idx for idx in range(self.__len__())], test_size=1 - train_ratio)
        return (Subset(self, train_idx), Subset(self, test_idx))
    
    
class SchemaMatchingDataset(Dataset):
    
    def __init__(self, schema_matches: pd.DataFrame, from_path=False):
        self.w2v = WordEmbedding(load_word_emb(w2v_config['data_dir'], 
                                               w2v_config['word2idx_path'],
                                               w2v_config['usedwordemb_path']))

        self.x_raw, self.x, self.y = [], [], []
        if not from_path:
            self._load_from_df(schema_matches)
            
    def _load_from_df(self, schema_matches):
        for idx, row in tqdm(schema_matches.iterrows(), total=len(schema_matches)):
            source_col, input_cols, target_col = row
            source_col, target_col = source_col.lower(), target_col.lower()
            input_cols = ';'.join([col.lower() for col in input_cols])
            input_sequence = "<BEG>;{};<SEQ>;{};<END>".format(source_col, input_cols)
            input_sequence_tok = input_sequence.split(';')
            end_pointer = input_sequence_tok.index('<END>') # basically len(input_sequence_tok)
            padding =  [end_pointer for _ in range(len(input_sequence_tok)-1)]
            embeddings = []
            try:
                target_pointer = torch.Tensor([input_sequence_tok.index(target_col)] + padding).long()
                # e.g. tokens = ['<BEG>', 'area (km 2 )', '<SEQ>', 'dvd title',  'region 2', 'region 1 (us)', '<END>']

                for token in input_sequence_tok:
                    embedding = np.mean([self.w2v(word) for word in token], axis=0)
                    embeddings.append(embedding)
                embeddings = torch.Tensor(embeddings)
                
                self.x.append(embeddings)
                self.x_raw.append(input_sequence)
                self.y.append(target_pointer)
            except:
                continue
               
    def __len__(self):
        return len(self.x_raw)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.x_raw[idx]
    
    def save(self, path):
        data = {
            'x': self.x,
            'x_raw': self.x_raw,
            'y': self.y
        }
        torch.save(data, os.path.join(path, 'schema_matching.pt'))
    
    def load(self, path):
        data = torch.load(os.path.join(path, 'schema_matching.pt'))
        x, x_raw, y = data['x'], data['x_raw'], data['y']
        self.x.extend(x)
        self.x_raw.extend(x_raw)
        self.y.extend(y)
        
    def split(self, train_ratio):
        train_idx, test_idx = train_test_split([idx for idx in range(self.__len__())], test_size=1 - train_ratio)
        return (Subset(self, train_idx), Subset(self, test_idx))
        
#class AlphabetSortingDataset(Dataset):
#    
#    def __init__(self, num_samples, min_len=20, max_len=60, alphabet='abcdefghijklmnopqrstuvwxyz'):
#        w2v = WordEmbedding(load_word_emb(w2v_config['data_dir'], 
#                                          w2v_config['word2idx_path'],
#                                          w2v_config['usedwordemb_path'])
#                           )
#        
#        self.num_samples = num_samples
#        self.alphabet = alphabet
#        self.embedding = {char: w2v(char) for char in self.alphabet}
#        self.min_len = min_len
#        self.max_len = max_len
#        self.x, self.chars, self.y = self._build()
#        
#    def _build(self):
#        array_len = torch.randint(low=self.min_len, 
#                            high=self.max_len + 1,
#                            size=(1,))
#        x_idxs = torch.randint(high=len(self.alphabet), size=(self.num_samples, array_len))
#        y = x_idxs.argsort()
#        x, chars = [], []
#        for idxs in x_idxs:
#            x_appendix, chars_appendix = [], []
#            for idx in idxs:
#                char = self.alphabet[idx]
#                chars_appendix.append(char)
#                x_appendix.append(self.embedding[char])
#            x.append(x_appendix)
#            chars.append(chars_appendix)
#        return torch.Tensor(x), chars, y
#    
#    def __len__(self):
#        return len(self.x)
#    
#    def __getitem__(self, idx):
#        return self.x[idx], self.y[idx], self.chars[idx]
        
    
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
        return self.x[idx], self.y[idx], self.x[idx]


class ExtendedWikiSQL(Dataset):

    def __init__(self):
        self.inputs, self.targets = [], []

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input': self.inputs[idx],
            'target': self.targets[idx]
        }

    def load_from_torch(self, path):
        self.inputs = torch.load('{}_inputs.pt'.format(path))
        self.targets = torch.load('{}_targets.pt'.format(path))