import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class SimpleEncoder(nn.Module):
    def __init__(self,n_words,n_factors,n_hidden):
        super().__init__()
        self.e = nn.Embedding(n_words,n_factors)
        self.lstm = nn.LSTM(n_factors,n_hidden)
        
    def forward(self,inp):
        out = self.e(inp)
        _,hidd = self.lstm(out)
        return hidd

class SimpleEncoderVLS(nn.Module):
    def __init__(self,n_words,n_factors,n_hidden,pad_id):
        super().__init__()
        self.e = nn.Embedding(n_words,n_factors)
        self.lstm = nn.LSTM(n_factors,n_hidden)
        self.pad_id = pad_id
        
    def forward(self,inp,ls):
        out = self.e(inp)
        out = pack_padded_sequence(out,ls)
        _,hidd = self.lstm(out)
        return hidd

class SimpleDecoder(nn.Module):
    def __init__(self,n_words,n_factors,n_hidden):
        super().__init__()
        self.e = nn.Embedding(n_words,n_factors)
        self.lstm = nn.LSTM(n_factors,n_hidden)
        self.fc = nn.Linear(n_hidden,n_words)
        
    def forward(self,inp,hidd):
        out = self.e(inp)
        out,hidd = self.lstm(out,hidd)
        out = self.fc(out)
        out = F.log_softmax(out,dim=-1)
        return out,hidd