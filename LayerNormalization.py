import torch.nn as nn
import math
import torch

class LayerNormalization(nn.Module):
    
    def __init__(self, eps=10**-6):
        super().__init()__
        self.eps = eps
        self.alpha = nn.Parameters(torch.ones(1))
        self.bias = nn.Parameters(torch.zeros(1))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std - self.eps) + self.bias