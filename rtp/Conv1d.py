import torch

import torch.nn as nn

import rtp_core

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__()
    
        '''
        TODO: padding_mode (always zeros now)
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.register_buffer('weight', torch.empty((self.out_channels, self.in_channels, self.kernel_size)))
        self.register_buffer('bias', torch.empty((self.out_channels)))

    def forward(self, last_out, threshold):
        return rtp_core.rtp_conv1d(last_out, self.weight, self.stride, self.padding, self.dilation, threshold)

class DebugConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__()
    
        '''
        TODO: padding_mode (always zeros now)
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.register_buffer('weight', torch.empty((self.out_channels, self.in_channels, self.kernel_size)))
        self.register_buffer('bias', torch.empty((self.out_channels)))

    def forward(self, last_out, threshold, stats):
        c, pruned_in_training, pruned_in_inference, total = rtp_core.debug_rtp_conv1d(last_out, self.weight, self.stride, self.padding, self.dilation, threshold)

        stats['pruned_in_training'] += pruned_in_training
        stats['pruned_in_inference'] += pruned_in_inference
        stats['total'] += total

        return c

class NPConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__()
    
        '''
        TODO: padding_mode (always zeros now)
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.register_buffer('weight', torch.empty((self.out_channels, self.in_channels, self.kernel_size)))
        self.register_buffer('bias', torch.empty((self.out_channels)))

    def forward(self, last_out):
        return rtp_core.conv1d(last_out, self.weight, self.stride, self.padding, self.dilation)
