import torch.nn as nn
import math
import torch

class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        assert d_model % heads == 0, "d_model is not divided by heads" 
        self.d_k = d_model // heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(self, Q, K, V, mask, dropout:nn.Dropout):
        d_k = Q.shape[-1]
        K = K.transpose(-2, -1) # inverse K here ------ Q @ K.t
        
        attetion_scores = Q @ K / math.sqrt(d_k) # you can use torch.metmul(Q, K) that also can work.
        
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_scores = attention_scores.softmax(dim=-1)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ V), attention_scores
    
    def forward(self, q, k, v, mask):
        Q = self.w_q(q) # (batch_size, seq_len, d_model)
        K = self.w_k(k)
        V = self.w_v(v)
        
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, heads, d_k) -> (batch_size, heads, seq_len, d_k)
        Q = Q.view(Q.shape[0], Q.shape[1], self.heads, self.d_k).transpose(1, 2)
        K = K.view(K.shape[0], K.shape[1], self.heads, self.d_k).transpose(1, 2)
        V = V.view(V.shape[0], V.shape[1], self.heads, self.d_k).transpose(1, 2)
        
        # scaled dot-product attention in parallel
        x, attention_scores = MultiHeadAttention.attention(Q, K, V, mask, self.dropout)
        
        # concat(x): (batch_size, heads, seq_len, d_k) -> (batch_size, seq_len, heads, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_k * self.heads)
        
        # then linear w_o * x
        return self.w_o(x) 