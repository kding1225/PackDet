import torch
import torch.nn as nn
from core.layers import ConvBlock


class MontageBlock(torch.nn.Module):
    def __init__(self, montage_type, use_extra_features, tile_mode='box', device = 'cuda'):
        super(MontageBlock, self).__init__()
        self.use_extra_features = use_extra_features
        self.num_extra_strides = 3 if self.use_extra_features else 0
        self.montage_type = montage_type
        self.device = device
        self.tile_mode = tile_mode

        if self.montage_type == 'type-1':
            self.montage_mat_x = torch.FloatTensor([
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]]
            ).to(device)
            self.montage_mat_y = torch.FloatTensor([
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]]
            ).to(device)
        elif self.montage_type == 'type-2':
            self.montage_mat_x = torch.FloatTensor([
                [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, -1, 0, -1, -1, -1, -1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, -1, 0, -1, -1, 0, 0, -1, 0, -1, -1],
                [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0]]
            ).to(device)
            self.montage_mat_y = torch.FloatTensor([
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1],
                [0, 0, 0, 0, 0, 0, -1, 0, -1, -1, 0, 0, -1, 0, -1, -1],
                [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0]]
            ).to(device)
        else:
            raise NotImplementedError
        if not self.use_extra_features:
            self.montage_mat_x = self.montage_mat_x[:, :-6]
            self.montage_mat_y = self.montage_mat_y[:, :-6]

    def forward(self, features):
        """
        put all features Q3-Q7 in a large feature map to get the montage feature map
        """
        device = features[0].device

        sizes = torch.tensor([f.shape[-2:] for f in features]).to(device)
        N, C = features[0].shape[:2]
        mon_height, mon_width = sizes[0][0] + sizes[1][0], sizes[0][1]
        mon_feature = features[0].new_zeros(N, C, mon_height, mon_width)  # no-grad
        boxes = self.compute_montage_pos2(sizes)
        
        # copy features, locations and scales
        if self.tile_mode == 'box':
            mon_feature = self.put_features_by_boxes(mon_feature, features, boxes)
        elif self.tile_mode == 'index':
            indices, regioin_sizes = self.compute_1d_indices(boxes, mon_height, mon_width)
            mon_feature = self.put_features_by_indices(mon_feature, features, indices)
        else:
            raise NotImplementedError

        return mon_feature
    
    def compute_1d_indices(self, boxes, H, W):
        """
        compute the 1d indices of the boxes
        i,j -> i*W+j
        """
        indices = []
        sizes = []
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            xs = torch.arange(x0, x1, device=self.device)
            ys = torch.arange(y0, y1, device=self.device)
            x, y = torch.meshgrid(xs, ys)
            cur_indices = torch.cat([x.reshape(-1,1), y.reshape(-1,1)], dim=1)
            indices.append(cur_indices)
            sizes.append(cur_indices.shape[0])
        indices = torch.cat(indices, dim=0)
        indices = indices[:,0] + indices[:,1]*W
        return indices, sizes
    
    def put_features_by_boxes(self, features_combine, features, boxes):
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            features_combine[:, :, y0:y1, x0:x1] = features[i]
        
        return features_combine
    
    def put_features_by_indices(self, features_combine, features, indices):
        """
        put features of all boxes into the features_combine
        features_combine: N*C*H*W
        features: list[tensor]
        """
        N, C, H, W = features_combine.shape
        features = torch.cat([f.view(N,C,-1) for f in features], dim=2)
        features_combine = features_combine.view(N,C,-1)
        features_combine[:,:,indices] = features
        features_combine = features_combine.view(N, C, H, W)
        return features_combine
    
    def compute_montage_pos(self, sizes):
        A, B = sizes[:5,0][None], sizes[:5,1][None]
        A, B = A.float(), B.float()
        x = (torch.mm(B, self.montage_mat_x)).view(-1, 2)
        y = (torch.mm(A, self.montage_mat_y)).view(-1, 2)
        poses = torch.cat([x, y], dim=1)[:, [0, 2, 1, 3]]
        return poses.long()
    
    def compute_montage_pos2(self, sizes):
        a0, b0, a1, b1, a2, b2, a3, b3, a4, b4 = sizes[:5,:].view(-1)
        if self.montage_type == 'type-1':
            b12 = b1 + b2
            boxes = torch.tensor([
                [0, 0, b0, a0],
                [0, a0, b1, a0 + a1],
                [b1, a0, b12, a0 + a2],
                [b12, a0, b12 + b3, a0 + a3],
                [b12 + b3, a0, b12 + b3 + b4, a0 + a4]
            ]).to(self.device)
            if self.use_extra_features:
                a02 = a0 + a2
                boxes_ = torch.tensor([
                    [b1, a02, b12, a0 + a1],
                    [b12, a02, b12 + b3, a02 + a3],
                    [b12 + b3, a02, b12 + b3 + b4, a02 + a4]
                ]).to(self.device)
                boxes = torch.cat([boxes,boxes_], dim=0)
        elif self.montage_type == 'type-2':
            a01 = a0 + a1
            b23, a23 = b2 + b3, a2 + a3
            boxes = torch.tensor([
                [0, 0, b0, a0],
                [0, a0, b1, a01],
                [b0 - b2, a01 - a2, b0, a01],
                [b0 - b23, a01 - a23, b0 - b2, a01 - a2],
                [b0 - b23 - b4, a01 - a23 - a4, b0 - b23, a01 - a23],
            ]).to(self.device)
            if self.use_extra_features:
                boxes_ = torch.tensor([
                    [b1, a01 - a2, b1 + b2, a01],
                    [b0 - b3, a01 - a23, b0, a01 - a2],
                    [b0 - b3 - b4, a01 - a23 - a4, b0 - b3, a01 - a23]
                ]).to(self.device)
                boxes = torch.cat([boxes,boxes_], dim=0)
        else:
            raise NotImplementedError
        return boxes.long()


class ConvBlockSeq(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1,
                 padding=1, stride=1, use_dcn=0, num_convs=1):
        super(ConvBlockSeq, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]*num_convs
        else:
            num_convs = len(kernel_size)

        if isinstance(padding, int):
            padding = [padding]*num_convs

        if isinstance(stride, int):
            stride = [stride]*num_convs

        if isinstance(use_dcn, (int, bool)):
            use_dcn = [use_dcn]*num_convs

        channels1 = [in_channels] + [out_channels]*(num_convs-1)
        channels2 = [out_channels]*num_convs

        self.conv_blocks = nn.Sequential(*[
            ConvBlock(c1, c2, k, p, s, dcn)
            for c1, c2, k, p, s, dcn in zip(channels1, channels2, kernel_size, padding, stride, use_dcn)
        ])

    def forward(self, x):
        return self.conv_blocks(x)


class MontageConvSeq(torch.nn.Module):
    """
    montage + convs
    """
    def __init__(self, in_channels, num_convs, kernel_size, enable_montage=True, use_dcn=False):
        super(MontageConvSeq, self).__init__()
        self.enable_montage = enable_montage
        if enable_montage:
            self.montage = MontageBlock('type-1', True)
        self.convs = ConvBlockSeq(
            in_channels, in_channels, 
            kernel_size=kernel_size, 
            num_convs=num_convs,
            use_dcn=use_dcn
        )
    def forward(self, Xs):
        """Xs includes extra features when use_extra_features=True"""
        if self.enable_montage:
            X = self.montage(Xs)  # in this case, Xs should be a list[tensor]
        else:
            X = Xs
        out = self.convs(X)
        return out


class SeparateConvSeq(torch.nn.Module):
    """
    apply convs to diff heads separately
    """
    def __init__(self, in_channels, num_convs, kernel_size, mul_streams=False, use_dcn=False):
        super(SeparateConvSeq, self).__init__()
        self.convs = ConvBlockSeq(
            in_channels, in_channels,
            kernel_size=kernel_size,
            num_convs=num_convs,
            use_dcn=use_dcn
        )
        self.mul_streams = mul_streams
        
    def forward(self, Xs):
        if self.mul_streams:
            out = self.forward_multiple_streams(Xs)
        else:
            out = self.forward_single_stream(Xs)
        return out
    
    def forward_single_stream(self, Xs):
        out = []
        for x in Xs:
            out.append(self.convs(x))
        return out
    
    def forward_multiple_streams(self, Xs):
        torch.cuda.synchronize()
        new_stream = torch.cuda.Stream()
        out = []
        for i, x in enumerate(Xs):
            if i < len(Xs)/2:
                out.append(self.convs(x))
            else:
                with torch.cuda.stream(new_stream):
                    out.append(self.convs(x))
        torch.cuda.synchronize()
        return out


def gen_problem(shapes, device):
    Xs = []
    for i, shape in enumerate(shapes):
        Xs.append(torch.rand(shape).to(device))
    return Xs


def timing(fun, params_list, n_repeat):
    
    # warm up
    for i in range(20):
        tmp = fun(*params_list)
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(n_repeat):
        tmp = fun(*params_list)
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    return start.elapsed_time(end)/n_repeat
