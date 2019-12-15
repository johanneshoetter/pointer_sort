import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import json
import os

class WordEmbedding(nn.Module):
    def __init__(self, word_emb):
        super(WordEmbedding, self).__init__()
        self.w2i, self.word_emb_val = word_emb
        
    def forward(self, word):
        emb = np.array(self.word_emb_val[self.w2i.get(word)])
        if emb.shape[0] != 300:
            emb = np.array(self.word_emb_val[self.w2i.get('_')]) # if not known, set placeholder
        return emb
    
def load_word_emb(data_dir, word2idx_path, usedwordemb_path):
    with open(os.path.join(data_dir, word2idx_path)) as inf:
        w2i = json.load(inf)
    with open(os.path.join(data_dir, usedwordemb_path), 'rb') as inf:
        word_emb_val = np.load(inf)
    return w2i, word_emb_val