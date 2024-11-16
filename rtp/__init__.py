# __package__ = 'rtp'

from rtp.Linear import Linear, DebugLinear, NPLinear

from rtp.Conv2d import Conv2d, DebugConv2d, NPConv2d

from rtp.Conv1d import Conv1d, DebugConv1d, NPConv1d

from rtp_core import linear, rtp_linear, debug_rtp_linear, conv2d, rtp_conv2d, debug_rtp_conv2d, conv1d, rtp_conv1d, debug_rtp_conv1d

# class NoPruneConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1):
#         super().__init__()
    
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.register_buffer('weight', torch.empty((self.out_features, self.in_features)))
#         self.register_buffer('bias', torch.empty((self.out_features)))
#         # 128 is currently the minimum `tile_n`, hence it gives the maximum workspace size; 16 is the default `max_par`
#         # self.register_buffer('workspace', torch.zeros(self.n // 128 * 16, dtype=torch.int), persistent=False)

#     def forward(self, last_out):
#         return rtp_core.no_prune_mat_mul(last_out, self.weight)