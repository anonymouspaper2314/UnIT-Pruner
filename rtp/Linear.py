import torch

import torch.nn as nn

import rtp_core

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
    
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.empty((self.out_features, self.in_features)))
        self.register_buffer('bias', torch.empty((self.out_features)))

    def forward(self, last_out, threshold):
        return rtp_core.rtp_linear(last_out, self.weight, threshold)

class DebugLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
    
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.empty((self.out_features, self.in_features)))
        self.register_buffer('bias', torch.empty((self.out_features)))

    def forward(self, last_out, threshold, stats):
        c, pruned_in_training, pruned_in_inference, total = rtp_core.debug_rtp_linear(last_out, self.weight, threshold)

        stats['pruned_in_training'] += pruned_in_training
        stats['pruned_in_inference'] += pruned_in_inference
        stats['total'] += total

        return c

class NPLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
    
        self.in_features = in_features
        self.out_features = out_features
        
        self.register_buffer('weight', torch.empty((self.out_features, self.in_features)))
        self.register_buffer('bias', torch.empty((self.out_features)))

    def forward(self, last_out):
        return rtp_core.linear(last_out, self.weight)