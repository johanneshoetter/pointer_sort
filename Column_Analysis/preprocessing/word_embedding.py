import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import json
import numpy as np
import os

class WordEmbeddingLite(nn.Module):
    def __init__(self, word_emb, ):
        super(WordEmbeddingLite, self).__init__()
        self.w2i, self.word_emb_val = word_emb
        
    def forward(self, word):
        return np.array(self.word_emb_val[self.w2i.get(word)])
    
def load_word_emb(data_dir, word2idx_path, usedwordemb_path):
    with open(os.path.join(data_dir, word2idx_path)) as inf:
            #r'C:\Users\Johannes\Desktop\Bachelorarbeit\40_Development\00_Service\Seq2SQL\dataservice\glove\word2idx.json') as inf:
        w2i = json.load(inf)
    with open(os.path.join(data_dir, usedwordemb_path), 'rb') as inf:
            #r'C:\Users\Johannes\Desktop\Bachelorarbeit\40_Development\00_Service\Seq2SQL\dataservice\glove\usedwordemb.npy',
            #'rb') as inf:
        word_emb_val = np.load(inf)
    return w2i, word_emb_val

#embedding = WordEmbeddingLite(word_emb)
#print(embedding("borderlands"))