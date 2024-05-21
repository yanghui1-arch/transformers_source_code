import torch.nn as nn
import math
import torch

class FeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff, dropout:float):
        super().__init__()
        # input and output is d_model. 
        self.linear_1 = nn.Linear(d_model, d_ff) # x * W1 + b1
        self.linear_2 = nn.Linear(d_ff, d_model) # max * W2 + b2
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # according to the last sentence we need to do following things:
        #            input          ->       inner-layer      ->        output
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))