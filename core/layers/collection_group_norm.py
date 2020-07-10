import torch.nn as nn
import torch
from torch.nn import init
import torch.nn.functional as F


class CollectionGroupNorm(nn.Module):

    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine', 'weight',
                     'bias']

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(CollectionGroupNorm, self).__init__()

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, inputs):
        """
        inputs: b*c*h*w for each tensor
        """
        b, c = inputs[0].shape[:2]
        shapes = [x.shape[2:] for x in inputs]
        inputs_reshaped = []
        for x, s in zip(inputs, shapes):
            input_reshaped = x.contiguous().view(b, c, s[0]*s[1], 1)
            inputs_reshaped.append(input_reshaped)
        inputs_reshaped = torch.cat(inputs_reshaped, dim=2)

        outs = F.group_norm(
            inputs_reshaped, self.num_groups, self.weight, self.bias, self.eps)

        outs = torch.split(outs, [s[0] * s[1] for s in shapes], dim=2)
        outs = [out.view(b, c, s[0], s[1]) for s, out in zip(shapes, outs)]

        return outs

    def extra_repr(self):
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)


if __name__ == '__main__':
    A = torch.ones([2, 32, 10, 10])
    B = torch.zeros([2, 32, 5, 5])
    C = torch.ones([2, 32, 2, 2]) * 4
    inputs = [A, B, C]

    cgn = CollectionGroupNorm(8, 32)
    outs = cgn(inputs)
    print(outs[0].shape, outs[1].shape, outs[2].shape)