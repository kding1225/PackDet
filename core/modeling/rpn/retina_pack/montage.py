import itertools
import torch
import torch.nn.functional as F
from torch import nn
from core.layers import ConvBlock, NoopLayer
from core.utils.registry import Registry
from core.modeling.rpn.utils import meshgrid

MONTAGE_BOXES = Registry()

# register levels
MONTAGE_LEVELS = {
    'type-1': [0, 1, 2, 3, 4],
    'type-2': [0, 1, 2, 3, 4, 2, 3, 4],
    'type-3': [0, 1, 2, 3, 4],
    'type-4': [0, 1, 2, 3, 4, 2, 3, 4],
    'type-5': [0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4],
    'type-6': [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4],
    'type-7': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4],
    'type-8': [0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4],
    'type-9': [0, 1, 2, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
    'type-stair1': [1, 1, 1, 1],
    'type-stair2': [2, 2, 2, 2],
    'type-stair3': [3, 3, 3, 3],
    'type-stair4': [4, 4, 4, 4],
    'type-sbs1': [1, 1, 1, 1],
    'type-sbs2': [2, 2, 2, 2],
    'type-sbs3': [3, 3, 3, 3],
    'type-sbs4': [4, 4, 4, 4],
    'type-grid1': [1, 1, 1, 1],
    'type-grid2': [2, 2, 2, 2],
    'type-grid3': [3, 3, 3, 3],
    'type-grid4': [4, 4, 4, 4]
}


class BasicMontageBlock(torch.nn.Module):
    """
    montage block to pack features from different scales
    """
    def __init__(self, fpn_strides, montage_box, montage_levels, device='cpu'):
        super(BasicMontageBlock, self).__init__()
        self.fpn_strides = fpn_strides
        self.device = device
        self.montage_box = montage_box
        self.levels = montage_levels

    def forward(self, features, sizes, visualizer=None):
        """
        put all features P3-P7 in a large feature map to get the montage feature map
        """
        N, C = features[0].shape[:2]
        boxes, mon_height, mon_width = self.montage_box(sizes, self.device)

        mon_feature = features[0].new_zeros(N, C, mon_height, mon_width)  # no-grad
        locations_map = -features[0].new_ones((mon_height, mon_width, 2), dtype=torch.float)
        scales_map = features[0].new_zeros((mon_height, mon_width), dtype=torch.long)

        # copy features, locations and scales
        all_locations = self.compute_locations(sizes)
        for i, (level, feat, box, loc) in enumerate(zip(self.levels, features, boxes, all_locations)):
            x0, y0, x1, y1 = box
            mon_feature[..., y0:y1, x0:x1] = feat
            locations_map[y0:y1, x0:x1, :] = loc
            scales_map[y0:y1, x0:x1] = level

        montage_info = dict(
            locations_map=locations_map,
            ranges=boxes,
            scales_map=scales_map,
            levels=self.levels
        )

        return mon_feature, montage_info

    def compute_locations(self, sizes):
        locations = []
        for l, siz in enumerate(sizes):
            h, w = siz
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[l],
                self.device
            )
            locations.append(locations_per_level)
        locations = [locations[l]+0.5*self.fpn_strides[l] for i, l in enumerate(self.levels)]
        return locations

    @staticmethod
    def compute_locations_per_level(h, w, stride, device):
        shift_y, shift_x = meshgrid(h, w, stride, device, dtype=torch.float32)
        locations = torch.cat([shift_x[..., None], shift_y[..., None]], dim=2)
        return locations


def plot_montage(boxes, height, width):

    import numpy as np
    import matplotlib.pyplot as plt

    boxes = boxes.cpu().numpy()
    scale_map = -np.ones((height, width), dtype=float)
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        scale_map[y0:y1, x0:x1] = i+1

    plt.figure(figsize=(15, 10))
    plt.imshow(scale_map)
    plt.show()
    # plt.savefig("scale_map.png")

    return scale_map


# *********************** montage boxes ****************************
# different methods to generate boxes, when adding a new montage kind
# please register it to MONTAGE_BOXES

def montage_pos_type12(sizes, device, use_extra_features=False):
    """
    type-1 and type-2 montage positioning, need image size divisible by 32
    sizes: list[(h,w)]
    """
    assert len(sizes) == 5
    a0, b0, a1, b1, a2, b2, a3, b3, a4, b4 = list(itertools.chain(*sizes))
    boxes = [
        [0, 0, b0, a0],
        [0, a0, b1, a0 + a1],
        [b1, a0, b1 + b2, a0 + a2],
        [b1 + b2, a0, b1 + b2 + b3, a0 + a3],
        [b1 + b2 + b3, a0, b1 + b2 + b3 + b4, a0 + a4]
    ]
    if use_extra_features:
        boxes.extend([
            [b1, a0 + a2, b1 + b2, a0 + a1],
            [b1 + b2, a0 + a2, b1 + b2 + b3, a0 + a2 + a3],
            [b1 + b2 + b3, a0 + a2, b1 + b2 + b3 + b4, a0 + a2 + a4]
        ])
    boxes = torch.tensor(boxes).to(device).long()
    mon_height, mon_width = a0 + a1, b0
    return boxes, mon_height, mon_width


@MONTAGE_BOXES.register("type-1")
def montage_pos_type1(sizes, device='cpu'):
    return montage_pos_type12(sizes, device, False)


@MONTAGE_BOXES.register("type-2")
def montage_pos_type2(sizes, device='cpu'):
    return montage_pos_type12(sizes, device, True)


def montage_pos_type34e_(sizes, device, use_extra_features=False):
    """
    type-3 and type-4 montage positioning, need image size divisible by 32
    sizes: list[(h,w)]
    """
    assert len(sizes) == 5
    a0, b0, a1, b1, a2, b2, a3, b3, a4, b4 = list(itertools.chain(*sizes))
    boxes = [
        [b0 - b2 * 2 - b1, a1 - a2 - a3 - a4, 2*b0 - b2 * 2 - b1, a0 + a1 - a2 - a3 - a4],  # 0
        [b0 - b2 * 2-b1, a0 + a1 - a2 - a3 - a4, b0 - b2 * 2, a0 + a1 - a2 - a3 - a4+a1],  # 1
        [b0 - b2, a0 + a1 - a2, b0, a0 + a1],  # 2
        [b0 - b2 - b3, a0 + a1 - a2 - a3, b0 - b2, a0 + a1 - a2],  # 3
        [b0 - b2 - b3 - b4, a0 + a1 - a2 - a3 - a4, b0 - b2 - b3, a0 + a1 - a2 - a3],  # 4
    ]
    if use_extra_features:
        boxes.extend([
            [b0 - b2 * 2, a0 + a1 - a2, b0 - b2, a0 + a1],  # 2'
            [b0 - b3, a0 + a1 - a2 - a3, b0, a0 + a1 - a2],  # 3'
            [b0 - b3 - b4, a0 + a1 - a2 - a3 - a4, b0 - b3, a0 + a1 - a2 - a3]  # 4'
        ])

    boxes = torch.tensor(boxes).to(device).long()
    xy_min = torch.min(boxes[:, :2], dim=0, keepdim=True)[0]
    boxes[:, :2] = boxes[:, :2] - xy_min
    boxes[:, 2:] = boxes[:, 2:] - xy_min

    mon_height, mon_width = torch.max(boxes[:, 3]), torch.max(boxes[:, 2])
    return boxes, mon_height, mon_width


def montage_pos_type34(sizes, device, use_extra_features=False):
    """
    type-3 and type-4 montage positioning, need image size divisible by 32
    sizes: list[(h,w)]
    """
    assert len(sizes) == 5
    a0, b0, a1, b1, a2, b2, a3, b3, a4, b4 = list(itertools.chain(*sizes))
    boxes = [
        [0, 0, b0, a0],
        [0, a0, b1, a0 + a1],
        [b0 - b2, a0 + a1 - a2, b0, a0 + a1],
        [b0 - b2 - b3, a0 + a1 - a2 - a3, b0 - b2, a0 + a1 - a2],
        [b0 - b2 - b3 - b4, a0 + a1 - a2 - a3 - a4, b0 - b2 - b3, a0 + a1 - a2 - a3],
    ]
    if use_extra_features:
        boxes.extend([
            [b1, a0 + a1 - a2, b1 + b2, a0 + a1],
            [b0 - b3, a0 + a1 - a2 - a3, b0, a0 + a1 - a2],
            [b0 - b3 - b4, a0 + a1 - a2 - a3 - a4, b0 - b3, a0 + a1 - a2 - a3]
        ])
    boxes = torch.tensor(boxes).to(device).long()
    mon_height, mon_width = a0 + a1, b0
    return boxes, mon_height, mon_width


@MONTAGE_BOXES.register("type-3")
def montage_pos_type3(sizes, device='cpu'):
    return montage_pos_type34(sizes, device, False)


@MONTAGE_BOXES.register("type-4")
def montage_pos_type4(sizes, device='cpu'):
    return montage_pos_type34(sizes, device, True)


@MONTAGE_BOXES.register("type-5")
def montage_pos_type5(sizes, device='cpu'):
    """
    type-5 montage positioning, need image size divisible by 128
    sizes: list[(h,w)]
    """
    assert len(sizes) == 5
    a0, b0, a1, b1, a2, b2, a3, b3, a4, b4 = list(itertools.chain(*sizes))
    boxes = [
        [0, 0, b0, a0],
        [0, a0, b1, a0+a1],
        [b1, a0, b1+b2, a0+a2],
        [b1+b2, a0, b0, a0+a2],
        [b1, a0+a2, b1+b2, a0+a1],
        [b1+b2, a0+a2, b0-b3, a0+a1-a3],
        [b0-b3, a0+a1-a2, b0, a0+a1-a3],
        [b0-b2, a0+a1-a3, b0-b3, a0+a1],
        [b0-b3, a0+a1-a3, b0-b4, a0+a1-a4],
        [b0-b4, a0+a1-a3, b0, a0+a1-a4],
        [b0-b3, a0+a1-a4, b0-b4, a0+a1],
        [b0-b4, a0+a1-a4, b0, a0+a1]
    ]
    boxes = torch.tensor(boxes).to(device).long()
    mon_height, mon_width = a0 + a1, b0
    return boxes, mon_height, mon_width


@MONTAGE_BOXES.register("type-6")
def montage_pos_type6(sizes, device='cpu'):
    """
    type-6 montage positioning, need image size divisible by 128
    sizes: list[(h,w)]
    """
    assert len(sizes) == 5
    a0, b0, a1, b1, a2, b2, a3, b3, a4, b4 = list(itertools.chain(*sizes))
    a0_mul2 = a0*2
    boxes = [
        [0, 0, b0, a0],
        [0, a0, b1, a0+a1],
        [b1, a0, b0, a0+a1],
        [0, a0+a1, b1, a0_mul2],
        [b1, a0+a1, b1 + b2, a0 + a1 + a2],
        [b1 + b2, a0+a1, b0, a0 + a1 + a2],
        [b1, a0 + a1 + a2, b1 + b2, a0_mul2],
        [b1 + b2, a0 + a1 + a2, b0 - b3, a0_mul2 - a3],
        [b0 - b3, a0_mul2 - a2, b0, a0_mul2 - a3],
        [b0 - b2, a0_mul2 - a3, b0 - b3, a0_mul2],
        [b0 - b3, a0_mul2 - a3, b0 - b4, a0_mul2 - a4],
        [b0 - b4, a0_mul2 - a3, b0, a0_mul2 - a4],
        [b0 - b3, a0_mul2 - a4, b0 - b4, a0_mul2],
        [b0 - b4, a0_mul2 - a4, b0, a0_mul2]
    ]
    boxes = torch.tensor(boxes).to(device).long()
    mon_height, mon_width = a0 + 2*a1, b0
    return boxes, mon_height, mon_width


@MONTAGE_BOXES.register("type-7")
def montage_pos_type7(sizes, device='cpu'):
    """
    type-7 montage positioning, need image size divisible by 128
    sizes: list[(h,w)]
    """
    assert len(sizes) == 5
    a0, b0, a1, b1, a2, b2, a3, b3, a4, b4 = list(itertools.chain(*sizes))
    a0_mul2, b0_mul2 = a0*2, b0*2
    a01, b01 = a0 + a1, b0 + b1
    boxes = [
        [0, 0, b0, a0],
        [b0, 0, b0_mul2, a0],
        [0, a0, b0, 2*a0],
        [b0, a0, b01, a01],
        [b01, a0, b0_mul2, a01],
        [b0, a01, b01, a0_mul2],
        [b01, a01, b01 + b2, a01 + a2],
        [b01 + b2, a01, b0_mul2, a01 + a2],
        [b01, a01 + a2, b01 + b2, a0_mul2],
        [b01 + b2, a01 + a2, b0_mul2 - b3, a0_mul2 - a3],
        [b0_mul2 - b3, a0_mul2 - a2, b0_mul2, a0_mul2 - a3],
        [b0_mul2 - b2, a0_mul2 - a3, b0_mul2 - b3, a0_mul2],
        [b0_mul2 - b3, a0_mul2 - a3, b0_mul2 - b4, a0_mul2 - a4],
        [b0_mul2 - b4, a0_mul2 - a3, b0_mul2, a0_mul2 - a4],
        [b0_mul2 - b3, a0_mul2 - a4, b0_mul2 - b4, a0_mul2],
        [b0_mul2 - b4, a0_mul2 - a4, b0_mul2, a0_mul2]
    ]
    boxes = torch.tensor(boxes).to(device).long()
    mon_height, mon_width = 2*a0, 2*b0
    return boxes, mon_height, mon_width


@MONTAGE_BOXES.register("type-8")
def montage_pos_type8(sizes, device='cpu'):
    """
    type-8, it only requires image size being divisible by 32
    sizes: list[(h,w)]
    """
    assert len(sizes) == 5
    a0, b0, a1, b1, a2, b2, a3, b3, a4, b4 = list(itertools.chain(*sizes))
    s2 = (b0-3*b2)//3
    s3 = (b1-3*b3)//2
    s4 = (b1-6*b4)//5

    def spacing_boxes(x0, y0, a, b, s, n):
        boxes = []
        for i in range(n):
            boxes.append([x0+i*b+i*s, y0, x0+(i+1)*b+i*s, y0+a])
        return boxes

    boxes = [
        [0, 0, b0, a0],
        [b0, 0, b0+b1, a1],
        [b0, a1, b0+b1, a0]
    ]
    boxes.extend(spacing_boxes(0, a0, a2, b2, s2, 3))
    boxes.extend(spacing_boxes(b0, a0, a3, b3, s3, 3))
    boxes.extend(spacing_boxes(b0, a0+a2-a4, a4, b4, s4, 6))
    boxes = torch.tensor(boxes).to(device).long()
    mon_height, mon_width = a0 + a2, b0 + b1
    return boxes, mon_height, mon_width


@MONTAGE_BOXES.register("type-9")
def montage_pos_type9(sizes, device='cpu'):
    """
    type-9, it only requires image size being divisible by 32
    sizes: list[(h,w)]
    """
    assert len(sizes) == 5
    a0, b0, a1, b1, a2, b2, a3, b3, a4, b4 = list(itertools.chain(*sizes))
    a3t = a3//2
    a3b = a3 - a3t
    b3l = b3//2
    b3r = b3 - b3l
    a4t = a4 // 2
    a4b = a4 - a4t
    b4l = b4 // 2
    b4r = b4 - b4l

    boxes = [
        [0, 0, b0, a0],  # 0
        [b1, a0, b0, a0+a1],  # 1
        [0, a0+a2, b2, a0+a1],  # 2
        [b2, a0+a1, b1, a0+a1+a2],  # 2
        [b2, a0+a2-a3, b2+b3, a0+a2],  # 3
        [b2+b3, a0+a2-a3-a4, b2+b3+b4, a0+a2-a3],  # 4
        [b2//2-b3l, a0+a2//2-a3t, b2//2+b3r, a0+a2//2+a3b],  # 3
        [b2+b2//2-b4l, a0+a2+a2//2-a4t, b2+b2//2+b4r, a0+a2+a2//2+a4b],  # 4
        [0, a0+a1+a2-a3, b3, a0+a1+a2],  # 3
        [b3, a0+a1+a2-a3-a4, b3+b4, a0+a1+a2-a3],  # 4
        [b0-b3, a0+a1+a2-a3, b0, a0+a1+a2],  # 3
        [b0-b3-b4, a0+a1+a2-a3-a4, b0-b3, a0+a1+a2-a3],  # 4
        [b0-b2-b3, a0+a1+a2-a3, b0-b2, a0+a1+a2],  # 3
        [b0-b2-b3-b4, a0+a1+a2-a3-a4, b0-b2-b3, a0+a1+a2-a3],  # 4
    ]
    boxes = torch.tensor(boxes).to(device).long()
    mon_height, mon_width = a0+a1+a2, b0
    return boxes, mon_height, mon_width


def montage_pos_stair(sizes, alpha, scale_id, device='cpu'):
    assert len(sizes) == 5
    a0, b0 = sizes[0]
    a, b = sizes[scale_id]
    mon_height, mon_width = int(a0*alpha), int(b0*alpha)
    cy, cx = mon_height//2, mon_width//2
    boxes = [
        [cx, cy-a, cx+b, cy],
        [cx+b, cy-2*a, cx+2*b, cy-a],
        [cx-b, cy, cx, cy+a],
        [cx-2*b, cy+a, cx-b, cy+2*a]
    ]
    boxes = torch.tensor(boxes).to(device).long()
    return boxes, mon_height, mon_width

@MONTAGE_BOXES.register("type-stair1")
def montage_pos_stair1(sizes, device='cpu'):
    return montage_pos_stair(sizes, 2, 1, device)

@MONTAGE_BOXES.register("type-stair2")
def montage_pos_stair2(sizes, device='cpu'):
    return montage_pos_stair(sizes, 1, 2, device)

@MONTAGE_BOXES.register("type-stair3")
def montage_pos_stair3(sizes, device='cpu'):
    return montage_pos_stair(sizes, 1, 3, device)

@MONTAGE_BOXES.register("type-stair4")
def montage_pos_stair4(sizes, device='cpu'):
    return montage_pos_stair(sizes, 1, 4, device)


def montage_pos_side_by_side(sizes, alpha, scale_id, device='cpu'):
    assert len(sizes) == 5
    a0, b0 = sizes[0]
    a, b = sizes[scale_id]
    mon_height, mon_width = int(a0*alpha), int(b0*alpha)
    cy, cx = mon_height//2, mon_width//2
    boxes = [
        [cx, cy, cx+b, cy+a],
        [cx+b, cy, cx+2*b, cy+a],
        [cx-b, cy, cx, cy+a],
        [cx-2*b, cy, cx-b, cy+a]
    ]
    boxes = torch.tensor(boxes).to(device).long()
    return boxes, mon_height, mon_width

@MONTAGE_BOXES.register("type-sbs1")
def montage_pos_side_by_side1(sizes, device='cpu'):
    return montage_pos_side_by_side(sizes, 2, 1, device)

@MONTAGE_BOXES.register("type-sbs2")
def montage_pos_side_by_side2(sizes, device='cpu'):
    return montage_pos_side_by_side(sizes, 1, 2, device)

@MONTAGE_BOXES.register("type-sbs3")
def montage_pos_side_by_side3(sizes, device='cpu'):
    return montage_pos_side_by_side(sizes, 1, 3, device)

@MONTAGE_BOXES.register("type-sbs4")
def montage_pos_side_by_side4(sizes, device='cpu'):
    return montage_pos_side_by_side(sizes, 1, 4, device)


def montage_pos_grid(sizes, alpha, scale_id, device='cpu'):
    assert len(sizes) == 5
    a0, b0 = sizes[0]
    a, b = sizes[scale_id]
    mon_height, mon_width = int(a0 * alpha), int(b0 * alpha)
    cy, cx = mon_height // 2, mon_width // 2
    boxes = [
        [cx-b, cy-a, cx, cy],
        [cx, cy-a, cx+b, cy],
        [cx-b, cy, cx, cy+a],
        [cx, cy, cx+b, cy+a]
    ]
    boxes = torch.tensor(boxes).to(device).long()
    return boxes, mon_height, mon_width

@MONTAGE_BOXES.register("type-grid1")
def montage_pos_grid1(sizes, device='cpu'):
    return montage_pos_grid(sizes, 2, 1, device)

@MONTAGE_BOXES.register("type-grid2")
def montage_pos_grid2(sizes, device='cpu'):
    return montage_pos_grid(sizes, 1, 2, device)

@MONTAGE_BOXES.register("type-grid3")
def montage_pos_grid3(sizes, device='cpu'):
    return montage_pos_grid(sizes, 1, 3, device)

@MONTAGE_BOXES.register("type-grid4")
def montage_pos_grid4(sizes, device='cpu'):
    return montage_pos_grid(sizes, 1, 4, device)


# ************************ montage features *********************
# methods to generate features include extra features

class MontageFeatLayer1234(nn.Module):
    """
    use conv-block to make extra features, each extra scale only occurs once
    """
    def __init__(self, in_channels, m, n):
        super(MontageFeatLayer1234, self).__init__()

        self.feat_expasions = nn.ModuleList(
            [NoopLayer()] * m + [
                ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
                for _ in range(n)
            ])

    def forward(self, features, levels):
        new_features = []
        for level, op in zip(levels, self.feat_expasions):
            new_features.append(op(features[level]))
        return new_features


class MontageFeatLayer567(nn.Module):
    """
    use 1*3/3*1/3*3 convs to make extra features
    """
    def __init__(self, in_channels, m, n):
        super(MontageFeatLayer567, self).__init__()
        module_list = [NoopLayer()]*m
        for i in range(n):
            module_list.extend(
                [
                    NoopLayer(),
                    ConvBlock(in_channels, in_channels, kernel_size=(1, 3), stride=1, padding=(0, 1)),
                    ConvBlock(in_channels, in_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),
                ]
            )
        module_list.append(
            ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        )
        self.feat_expasions = nn.ModuleList(module_list)

    def forward(self, features, levels):
        new_features = []
        for level, op in zip(levels, self.feat_expasions):
            new_features.append(op(features[level]))
        return new_features


class MontageFeatLayer567_3x3conv(nn.Module):
    """
    use 3x3 convs to make extra features
    """
    def __init__(self, in_channels, m, n):
        super(MontageFeatLayer567_3x3conv, self).__init__()
        module_list = [NoopLayer()]*m
        for i in range(n):
            module_list.extend(
                [
                    NoopLayer(),
                    ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                    ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                ]
            )
        module_list.append(
            ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        )
        self.feat_expasions = nn.ModuleList(module_list)

    def forward(self, features, levels):
        new_features = []
        level_prv = -1
        x_prv = None
        for level, op in zip(levels, self.feat_expasions):
            if level == level_prv:
                x = op(x_prv)
            else:
                x = op(features[level])
            x_prv = x
            new_features.append(x)
        return new_features


class MontageFeatLayer(nn.Module):
    def __init__(self, in_channels, levels, mode):
        super(MontageFeatLayer, self).__init__()
        self.mode = mode
        self.levels = levels
        assert mode in ['recursive', 'separate']

        uniques, scales_cnt = torch.unique(torch.tensor(levels).int(), sorted=True, return_counts=True)

        for i, n in zip(uniques, scales_cnt):
            for j in range(n):
                if j == 0:
                    layer = NoopLayer()
                else:
                    layer = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
                setattr(self, "mf_s%d_%d"%(i, j), layer)

        print(self)

    def forward(self, features):

        last_idx_dict = dict()
        if self.mode == 'recursive':
            last_features_dict = dict()

        new_features = []
        for l in self.levels:
            idx = last_idx_dict.get(l, -1)
            idx += 1
            last_idx_dict[l] = idx
            op = getattr(self, "mf_s%d_%d"%(l, idx))
            # print("mf_s%d_%d"%(l, idx))
            if self.mode == 'recursive':
                x = op(last_features_dict.get(l, features[l]))
                last_features_dict[l] = x
            else:
                x = op(features[l])
            new_features.append(x)

        return new_features, self.levels


# *********************** build functions ***********************

def build_montage_layer(cfg, in_channels):
    basic_montage_type = cfg.MODEL.RETINAPACK.BASIC_MONTAGE_TYPE
    montage_box = MONTAGE_BOXES[basic_montage_type]
    montage_levels = MONTAGE_LEVELS[basic_montage_type]
    fpn_strides = cfg.MODEL.RETINAPACK.FPN_STRIDES
    device = cfg.MODEL.DEVICE
    return BasicMontageBlock(fpn_strides, montage_box, montage_levels, device)


def build_montage_feat_layer(cfg, in_channels):
    basic_montage_type = cfg.MODEL.RETINAPACK.BASIC_MONTAGE_TYPE
    levels = MONTAGE_LEVELS[basic_montage_type]
    mode = cfg.MODEL.RETINAPACK.MONTAGE_FEAT_MODE
    return MontageFeatLayer(in_channels, levels, mode)