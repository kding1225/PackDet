import math
import torch.nn.functional as F
import torch
from torch import nn

from core.modeling.rpn.utils import normal_init
from .inference import make_packdet_postprocessor
from .inference2 import make_packdet_postprocessor2
from .loss import make_packdet_loss_evaluator
from .loss2 import make_packdet_loss_evaluator2
from core.layers import MonategScaleLayer, ConvBlock, CollectionConvBlock
from .feat_embed import build_feat_embed_layer
from .montage import build_montage_layer, build_montage_feat_layer
from core.modeling.rpn.utils import meshgrid


class Predict(torch.nn.Module):
    """
    predict different targets: cls, loc and centerness
    """
    def __init__(self, cfg, in_channels):
        super(Predict, self).__init__()
        device = cfg.MODEL.DEVICE
        num_classes = cfg.MODEL.PACKDET.NUM_CLASSES - 1
        self.norm_reg_targets = cfg.MODEL.PACKDET.NORM_REG_TARGETS
        self.use_dcn_in_tower = cfg.MODEL.PACKDET.USE_DCN_IN_TOWER
        self.fpn_strides = torch.FloatTensor(cfg.MODEL.PACKDET.FPN_STRIDES).to(device)
        self.centerness_on_reg = cfg.MODEL.PACKDET.CENTERNESS_ON_REG

        assert cfg.MODEL.PACKDET.PREDICT_LAYERS_GN_TYPE in ['gn', 'cgn']
        self.use_cgn = cfg.MODEL.PACKDET.PREDICT_LAYERS_GN_TYPE == 'cgn'
        conv_block = CollectionConvBlock if self.use_cgn else ConvBlock

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.PACKDET.NUM_CONVS):
            use_dcn_in_tower = True if self.use_dcn_in_tower and \
                        i == cfg.MODEL.PACKDET.NUM_CONVS - 1 else False
            cls_tower.append(
                conv_block(
                    in_channels, in_channels,
                    kernel_size=3, stride=1, padding=1,
                    use_dcn=use_dcn_in_tower
                )
            )
            bbox_tower.append(
                conv_block(
                    in_channels, in_channels,
                    kernel_size=3, stride=1, padding=1,
                    use_dcn=use_dcn_in_tower
                )
            )

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)

        # initialization
        for modules in [self.cls_logits, self.bbox_pred, self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    normal_init(l, mean=0, std=0.01, bias=0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.PACKDET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scale_layer = MonategScaleLayer(5)

    def forward(self, x, mode, **kwargs):
        assert mode in ['montage', 'no-montage']
        if mode == 'montage':
            return self.forward_montage(x, **kwargs)
        else:
            return self.forward_no_montage(x, **kwargs)

    def forward_no_montage(self, x, levels=None):
        """
        x: list[tensor]
        """
        logits = []
        bbox_reg = []
        centerness = []

        if self.use_cgn:  # forward layer by layer for all scales
            cls_towers = self.cls_tower(x)
            box_towers = self.bbox_tower(x)
        else:  # forward tower by tower
            cls_towers = [self.cls_tower(feature) for feature in x]
            box_towers = [self.bbox_tower(feature) for feature in x]

        for i, (level, feature) in enumerate(zip(levels, x)):
            cls_tower = cls_towers[i]
            box_tower = box_towers[i]

            logits.append(self.cls_logits(cls_tower))
            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))

            bbox_pred = self.scale_layer(self.bbox_pred(box_tower), level)
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred)
                bbox_reg.append(
                    bbox_pred if self.training else
                    bbox_pred * self.fpn_strides[level]
                )
            else:
                bbox_reg.append(torch.exp(bbox_pred))

        return logits, bbox_reg, centerness

    def forward_montage(self, mon_feature, scales_map=None):
        """
        Arguments:
            feature (tensor): montage feature
            scales_map (tensor[int]): scale indices of each pos in feature
        """
        cls_tower = self.cls_tower(mon_feature)
        box_tower = self.bbox_tower(mon_feature)

        # cls pred
        logits = self.cls_logits(cls_tower)

        # centerness pred
        if self.centerness_on_reg:
            centerness = self.centerness(box_tower)
        else:
            centerness = self.centerness(cls_tower)

        # box pred
        bbox_pred = self.bbox_pred(box_tower)
        bbox_pred = self.scale_layer(bbox_pred, scales_map)
        if self.norm_reg_targets:
            bbox_pred = F.relu(bbox_pred)  # assure bbox_reg>=0
            bbox_reg = bbox_pred if self.training else \
                bbox_pred * self.fpn_strides[scales_map]  # element-wise rescaling
        else:
            bbox_reg = torch.exp(bbox_pred)

        return logits, bbox_reg, centerness


class PACKDETHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(PACKDETHead, self).__init__()
        self.device = cfg.MODEL.DEVICE
        self.fpn_strides = cfg.MODEL.PACKDET.FPN_STRIDES
        self.train_enable_montage_speedup = cfg.MODEL.PACKDET.TRAIN_ENABLE_MONTAGE_SPEEDUP
        self.test_enable_montage_speedup = cfg.MODEL.PACKDET.TEST_ENABLE_MONTAGE_SPEEDUP

        self.mf = build_montage_feat_layer(cfg, in_channels)
        self.mb = build_montage_layer(cfg, in_channels)
        self.fe = build_feat_embed_layer(cfg, in_channels)
        out_channels = self.fe.out_channels
        self.predictor = Predict(cfg, out_channels)

    def forward(self, features):
        """
        features: P3-P7
        """
        sizes = [f.shape[-2:] for f in features]

        # generate montage feature
        features, levels = self.mf(features)

        enable_montage_speedup = self.train_enable_montage_speedup if \
            self.training else self.test_enable_montage_speedup

        if enable_montage_speedup:
            mon_feature, montage_info = self.mb(features, sizes)
            mon_feature = self.fe(mon_feature)

            # predict layers
            logits, bbox_reg, centerness = self.predictor(
                mon_feature, 'montage',
                scales_map=montage_info["scales_map"]
            )
        else:
            features = self.fe(features)

            logits, bbox_reg, centerness = self.predictor(
                features, 'no-montage', levels=levels
            )
            locations = self.compute_locations(sizes, levels)
            montage_info = {
                "levels": levels,
                "locations_map": [loc.view(-1, 2) for loc in locations]
            }

        return logits, bbox_reg, centerness, montage_info

    def compute_locations(self, sizes, levels):
        locations = []
        for l, siz in enumerate(sizes):
            h, w = siz
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[l],
                self.device
            )
            locations.append(locations_per_level)
        locations = [locations[l] for l in levels]
        return locations

    @staticmethod
    def compute_locations_per_level(h, w, stride, device):
        shift_y, shift_x = meshgrid(h, w, stride, device, dtype=torch.float32)
        locations = torch.cat([shift_x[..., None], shift_y[..., None]], dim=2) + stride // 2
        return locations


class PACKDETModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """
    def __init__(self, cfg, in_channels):
        super(PACKDETModule, self).__init__()
        self.norm_reg_targets = cfg.MODEL.PACKDET.NORM_REG_TARGETS
        self.fpn_strides = torch.FloatTensor(cfg.MODEL.PACKDET.FPN_STRIDES).to(cfg.MODEL.DEVICE)

        self.train_enable_montage_speedup = cfg.MODEL.PACKDET.TRAIN_ENABLE_MONTAGE_SPEEDUP
        self.test_enable_montage_speedup = cfg.MODEL.PACKDET.TEST_ENABLE_MONTAGE_SPEEDUP

        # for train
        if self.train_enable_montage_speedup:
            loss_evaluator = make_packdet_loss_evaluator2(cfg)
        else:
            loss_evaluator = make_packdet_loss_evaluator(cfg)

        # for test
        if self.test_enable_montage_speedup:
            box_selector_test = make_packdet_postprocessor2(cfg)
        else:
            box_selector_test = make_packdet_postprocessor(cfg)

        self.head = PACKDETHead(cfg, in_channels)
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None, visualizer=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """

        box_cls, box_regression, centerness, montage_info = \
            self.head(features)  # level first

        if self.training:
            return self._forward_train(
                montage_info, box_cls,
                box_regression,
                centerness, targets
            )
        else:
            return self._forward_test(
                montage_info, box_cls, box_regression,
                centerness, images.image_sizes
            )

    def _forward_train(self, montage_info, box_cls, box_regression,
                       centerness, targets):
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            montage_info, box_cls, box_regression, centerness, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }
        return None, losses

    def _forward_test(self, montage_info, box_cls, box_regression,
                      centerness, image_sizes):
        boxes = self.box_selector_test(
            montage_info, box_cls, box_regression,
            centerness, image_sizes
        )
        return boxes, {}


def build_packdet(cfg, in_channels):
    return PACKDETModule(cfg, in_channels)