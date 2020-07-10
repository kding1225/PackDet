import torch
from torch import nn


class MonategScaleLayer(nn.Module):
    def __init__(self, num, init_value=1.0):
        super(MonategScaleLayer, self).__init__()
        self.scales = nn.Parameter(torch.FloatTensor([init_value] * num))

    def forward(self, feature, scales_map):
        return feature*self.scales[scales_map]