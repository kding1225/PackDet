import torch
import torch.nn as nn
from core.layers import DFConv2d
from core.modeling.rpn.utils import normal_init


class ConvBlock(torch.nn.Module):
    """
    a single conv-bn/gn-relu
    use_dcn: use deformable conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=1,
                 padding=1, stride=1, use_dcn=False, dilation=1,
                 gn_groups=32, use_gn=True, use_act=True):

        super(ConvBlock, self).__init__()
        self.use_gn = use_gn
        self.use_act = use_act

        # Deformable convolution
        if use_dcn:
            conv_func = DFConv2d
        else:
            conv_func = nn.Conv2d

        self.conv = conv_func(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
                dilation=dilation
            )

        if use_gn:
            self.gn = nn.GroupNorm(gn_groups, out_channels)
        if use_act:
            self.relu = nn.ReLU(inplace=True)

        normal_init(self.conv, mean=0, std=0.01, bias=0)

    def forward(self, x):
        out = self.conv(x)
        if self.use_gn:
            out = self.gn(out)
        if self.use_act:
            out = self.relu(out)
        return out