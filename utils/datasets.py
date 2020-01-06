from torch.utils.data import Dataset
import torch
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
    
    def split(self, train_ratio, random_state=42):
        np.random.seed(42)
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
            source_col, input_cols, target_cols = row
            no_target = False if target_cols[0] != '<NONE>' else True
            source_col = source_col.lower()
            input_cols = ';'.join([col.lower() for col in input_cols])
            target_cols = [target_col.lower() for target_col in target_cols]
            input_sequence = "<BEG>;{};<SEQ>;{};<END>".format(source_col, input_cols)
            input_sequence_tok = input_sequence.split(';')
            end_pointer = input_sequence_tok.index('<END>') # basically len(input_sequence_tok)
            
            padding =  [end_pointer for _ in range(len(input_sequence_tok)-len(target_cols))]
            embeddings = []
            try:
                if no_target:
                    targets = [end_pointer]
                else:
                    targets = [torch.Tensor([input_sequence_tok.index(target_col)]) for target_col in target_cols]
                target_pointer =  torch.Tensor(targets + padding).long()
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
    
    def collate(self, batch):
        input_size = len(self.w2v('<BEG>'))
        x, y = [entry[0] for entry in batch], [entry[1] for entry in batch]
        max_len = max([len(sequence) for sequence in x])
        for idx, (sequence, true) in enumerate(zip(x, y)):
            len_padding = max_len - len(sequence)
            if len_padding > 0:
                target_padding = torch.Tensor([len(sequence) - 1 for _ in range(len_padding)]).long()
                sequence_padding = torch.zeros(len_padding, input_size)
                true = torch.cat([true, target_padding])
                sequence = torch.cat([sequence, sequence_padding])
                x[idx] = sequence
                y[idx] = true
        # make tensors out of list of tensors
        x = torch.cat([torch.unsqueeze(sequence, dim=0) for sequence in x], dim=0)
        y = torch.cat([torch.unsqueeze(targets, dim=0) for targets in y], dim=0)
        return x, y
    
    def save(self, path):
        data = {
            'x': self.x,
            'x_raw': self.x_raw,
            'y': self.y
        }
        torch.save(data, os.path.join(path, 'schema_matching_1toN.pt'))
    
    def load(self, path):
        data = torch.load(os.path.join(path, 'schema_matching_1toN.pt'))
        x, x_raw, y = data['x'], data['x_raw'], data['y']
        self.x.extend(x)
        self.x_raw.extend(x_raw)
        self.y.extend(y)
        
    def split(self, train_ratio, random_state=42):
        np.random.seed(random_state)
        train_idx, test_idx = train_test_split([idx for idx in range(self.__len__())], test_size=1 - train_ratio, random_state=random_state)
        return {
            'train': {
                'data': Subset(self, train_idx),
                'idxs': train_idx
            },
            'test': {
                'data': Subset(self, test_idx),
                'idxs': test_idx
            }
        }
    
    def yield_bootstrap(self, num_samples, batch_size, random_state=42, use_train=False, train_ratio=0.8, yield_idxs=False):
        np.random.seed(random_state)
        # get the indices, allow replace
        splits = self.split(train_ratio, random_state=random_state)
        if use_train:
            idxs_pool = splits['train']['idxs']
        else:
            idxs_pool = splits['test']['idxs']
        idxs_buffer = np.random.choice(idxs_pool, size=(num_samples, batch_size), replace=True)
        for idxs in idxs_buffer:
            if yield_idxs:
                yield idxs
            else:
                yield self.collate(Subset(self, idxs))
        # yield a Subset of the data on the indices
        
    def yield_bootstrap_by_class(self, num_samples, batch_size, random_state=42, use_train=False, train_ratio=0.8):
        is_1to0 = lambda x: len(x.unique()) == 1
        is_1to1 = lambda x: len(x.unique()) == 2
        is_1toN = lambda x: len(x.unique()) >= 3

        for idxs in self.yield_bootstrap(num_samples, batch_size, random_state=random_state, use_train=use_train, 
                                    train_ratio=train_ratio, yield_idxs=True):
            idxs_buffer_1to1, idxs_buffer_1toN, idxs_buffer_1to0 = [], [], []
            for idx in idxs:
                target = self.__getitem__(idx)[1]
                if is_1to1(target):
                    idxs_buffer_1to1.append(idx)
                elif is_1toN(target):
                    idxs_buffer_1toN.append(idx)
                elif is_1to0(target):
                    idxs_buffer_1to0.append(idx)
                else:
                    continue
            data = {
                '1to0': self.collate(Subset(self, idxs_buffer_1to0)),
                '1to1': self.collate(Subset(self, idxs_buffer_1to1)),
                '1toN': self.collate(Subset(self, idxs_buffer_1toN))
            }
            yield data
        