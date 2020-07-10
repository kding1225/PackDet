"""
This file contains specific functions for computing losses of FCOS
file
"""
import os
import torch
from torch import nn
from ..utils import concat_box_prediction_layers
from core.layers import IOULoss
from core.layers import SigmoidFocalLoss
from core.modeling.matcher import Matcher
from core.modeling.utils import cat
from core.structures.boxlist_ops import boxlist_iou
from core.structures.boxlist_ops import cat_boxlist


INF = 100000000
OBJECT_SIZES = [
    [-1, 64],
    [64, 128],
    [128, 256],
    [256, 512],
    [512, INF]
]


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


class PACKDETLossComputation(object):
    """
    This class computes the PACKDET losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.PACKDET.LOSS_GAMMA,
            cfg.MODEL.PACKDET.LOSS_ALPHA
        )
        self.device = cfg.MODEL.DEVICE
        self.fpn_strides = torch.FloatTensor(cfg.MODEL.PACKDET.FPN_STRIDES).to(self.device)
        self.center_sampling_radius = cfg.MODEL.PACKDET.CENTER_SAMPLING_RADIUS
        self.iou_loss_type = cfg.MODEL.PACKDET.IOU_LOSS_TYPE
        self.norm_reg_targets = cfg.MODEL.PACKDET.NORM_REG_TARGETS
        self.basic_montage_type = cfg.MODEL.PACKDET.BASIC_MONTAGE_TYPE

        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="none")

        self.object_sizes_of_interest = torch.FloatTensor(OBJECT_SIZES).to(self.device)

    def get_sample_region(self, gt, strides, xs, ys, radius=1.0):
        '''
        This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss2.py#L42
        '''
        num_gts = gt.shape[0]
        K = len(xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return xs.new_zeros(xs.shape, dtype=torch.uint8)

        strides = strides.view(-1, 1) * radius
        xmin = center_x - strides
        ymin = center_y - strides
        xmax = center_x + strides
        ymax = center_y + strides

        # limit sample region in gt
        center_gt[:, :, 0] = torch.where(
            xmin > gt[:, :, 0], xmin, gt[:, :, 0]
        )
        center_gt[:, :, 1] = torch.where(
            ymin > gt[:, :, 1], ymin, gt[:, :, 1]
        )
        center_gt[:, :, 2] = torch.where(
            xmax > gt[:, :, 2],
            gt[:, :, 2], xmax
        )
        center_gt[:, :, 3] = torch.where(
            ymax > gt[:, :, 3],
            gt[:, :, 3], ymax
        )

        left = xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - xs[:, None]
        top = ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

        return inside_gt_bbox_mask

    def prepare_targets(self, montage_info, targets):
        points = montage_info["locations_map"]
        scales_map = montage_info["scales_map"]

        H, W = scales_map.shape
        points = points.view(-1, 2)
        scales_map = scales_map.view(-1)

        expanded_object_sizes_of_interest = self.object_sizes_of_interest[scales_map, :]
        fpn_strides = self.fpn_strides[scales_map]

        labels, reg_targets = self.compute_targets_for_locations(
            points, targets, expanded_object_sizes_of_interest, fpn_strides
        )  # labels and reg_targets are image first

        labels = torch.cat(labels, dim=0)
        reg_targets = torch.cat(reg_targets, dim=0)

        if self.norm_reg_targets:
            reg_targets = reg_targets/fpn_strides.view(-1,1).repeat(len(targets), 1)

        return labels, reg_targets

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest,
                                      fpn_strides):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]  # (H*W)*(#boxes)
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)  # (H*W)*(#boxes)*4

            if self.center_sampling_radius > 0:
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    fpn_strides,
                    xs, ys,
                    radius=self.center_sampling_radius
                )
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0  # (H*W)*(#boxes)

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)  # (H*W)*(#boxes)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            # each point is assigned to a box
            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]  # (H*W)*4
            labels_per_im = labels_per_im[locations_to_gt_inds]  # get label of each pos, (H*W,)
            labels_per_im[locations_to_min_area == INF] = 0  # background

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    @staticmethod
    def compute_centerness_targets(reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, montage_info, box_cls, box_regression, centerness, targets):
        """
        Arguments:
            locations (Tensor)
            box_cls (Tensor)
            box_regression (Tensor)
            centerness (Tensor)
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        N, num_classes = box_cls.shape[:2]

        labels_flatten, reg_targets_flatten = self.prepare_targets(montage_info, targets)

        box_cls_flatten = box_cls.permute(0, 2, 3, 1).reshape(-1, num_classes)
        box_regression_flatten = box_regression.permute(0, 2, 3, 1).reshape(-1, 4)
        centerness_flatten = centerness.reshape(-1)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int(),
            None,
            None
        ) / num_pos_avg_per_gpu

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)

            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
            centerness_loss = centerness_loss.sum() / num_pos_avg_per_gpu

            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
            sum_centerness_targets_avg_per_gpu = \
                reduce_sum(centerness_targets.sum()).item() / float(num_gpus)

            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            ) / sum_centerness_targets_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum()
            reduce_sum(centerness_flatten.new_tensor([0.0]))
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss


def make_packdet_loss_evaluator2(cfg):
    loss_evaluator = PACKDETLossComputation(cfg)
    return loss_evaluator
