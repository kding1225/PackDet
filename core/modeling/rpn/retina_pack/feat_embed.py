import torch
import torch.nn.functional as F
from torch import nn

from core.layers import ConvBlock, CollectionConvBlock
from core.utils.registry import Registry


# ATTENTION: all FE modules should provide attribute: `out_channels`
FE_MODULES = Registry()


class FEModule(torch.nn.Module):
    """
    Feature embedding module
    """
    def __init__(self, cfg, in_channels):
        super(FEModule, self).__init__()

        num_convs = cfg.MODEL.RETINAPACK.FE.NUM_CONVS
        self.out_channels = cfg.MODEL.RETINAPACK.FE.OUT_CHANNELS
        self.reduction = ConvBlock(
            in_channels, self.out_channels,
            kernel_size=1, stride=1, padding=0
        )
        self.conv_blocks = nn.ModuleList([
            ConvBlock(self.out_channels,
                      self.out_channels,
                      kernel_size=3,
                      use_dcn=False,
                      use_gn=True,
                      stride=1,
                      padding=1,
                      use_act=True)
            for _ in range(num_convs)
        ])

    def forward(self, xs):
        if isinstance(xs, list):
            out = [self._forward(x) for x in xs]
        else:
            out = self._forward(xs)
        return out

    def _forward(self, x):
        x = self.reduction(x)
        tmp = x
        for l, conv in enumerate(self.conv_blocks):
            tmp = conv(tmp) + x
        return tmp


class CgnFEModule(torch.nn.Module):
    """
    Feature embedding module using cgn
    """
    def __init__(self, cfg, in_channels):
        super(CgnFEModule, self).__init__()

        num_convs = cfg.MODEL.RETINAPACK.CGN_FE.NUM_CONVS
        self.out_channels = cfg.MODEL.RETINAPACK.CGN_FE.OUT_CHANNELS

        self.reduction = CollectionConvBlock(
            in_channels, self.out_channels,
            kernel_size=1, stride=1, padding=0
        )
        self.conv_blocks = nn.ModuleList([
            CollectionConvBlock(self.out_channels,
                                self.out_channels,
                                kernel_size=3,
                                use_dcn=False,
                                use_gn=True,
                                stride=1,
                                padding=1,
                                use_act=True)
            for _ in range(num_convs)
        ])

    def forward(self, xs):
        xs = self.reduction(xs)
        tmp = xs
        for l, conv in enumerate(self.conv_blocks):
            tmp = conv(tmp)
            tmp = [t + x for t, x in zip(tmp, xs)]
        return tmp


@FE_MODULES.register("FE")
def build_fe_module_fe(cfg, in_channels):
    return FEModule(cfg, in_channels)


@FE_MODULES.register("CGN_FE")
def build_fe_module_cgn_fe(cfg, in_channels):
    return CgnFEModule(cfg, in_channels)


def build_feat_embed_layer(cfg, in_channels):
    feat_embed_type = cfg.MODEL.RETINAPACK.FE_TYPE
    return FE_MODULES[feat_embed_type](cfg, in_channels)