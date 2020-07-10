import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_retinanet_postprocessor
from .loss import make_retinanet_loss_evaluator
from .montage import build_montage_feat_layer, build_montage_layer, MONTAGE_LEVELS
from .feat_embed import build_feat_embed_layer
from ..anchor_generator import make_anchor_generator_retinanet

from core.modeling.box_coder import BoxCoder


class RetinaNetHead(torch.nn.Module):
    """
    Adds a RetinNet head with classification and regression heads
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RetinaNetHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.RETINAPACK.NUM_CLASSES - 1
        num_anchors = len(cfg.MODEL.RETINAPACK.ASPECT_RATIOS) \
                      * cfg.MODEL.RETINAPACK.SCALES_PER_OCTAVE

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.RETINAPACK.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, stride=1,
            padding=1
        )

        # Initialization
        for modules in [self.cls_tower, self.bbox_tower, self.cls_logits,
                        self.bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # retinanet_bias_init
        prior_prob = cfg.MODEL.RETINAPACK.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, montage_feature, montage_info):
        logits = self.cls_logits(self.cls_tower(montage_feature))
        bbox_reg = self.bbox_pred(self.bbox_tower(montage_feature))
        return self._split(logits, bbox_reg, montage_info)

    def _split(self, logits, bbox_reg, montage_info):
        ranges = montage_info['ranges']
        logits_list = []
        bbox_reg_list = []
        for box in ranges:
            xmin, ymin, xmax, ymax = box
            logits_list.append(logits[:, :, ymin:ymax, xmin:xmax].contiguous())
            bbox_reg_list.append(bbox_reg[:, :, ymin:ymax, xmin:xmax].contiguous())
        return logits_list, bbox_reg_list


class RetinaNetModule(torch.nn.Module):
    """
    Module for RetinaNet computation. Takes feature maps from the backbone and
    RetinaNet outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(RetinaNetModule, self).__init__()

        self.cfg = cfg.clone()

        self.mf = build_montage_feat_layer(cfg, in_channels)
        self.mb = build_montage_layer(cfg, in_channels)
        self.fe = build_feat_embed_layer(cfg, in_channels)

        # modify default anchor generator
        levels = MONTAGE_LEVELS[cfg.MODEL.RETINAPACK.BASIC_MONTAGE_TYPE]
        anchor_sizes = cfg.MODEL.RETINAPACK.ANCHOR_SIZES
        anchor_sizes = [anchor_sizes[i] for i in levels]
        anchor_strides = cfg.MODEL.RETINAPACK.ANCHOR_STRIDES
        anchor_strides = [anchor_strides[i] for i in levels]
        anchor_generator = make_anchor_generator_retinanet(cfg, anchor_sizes, anchor_strides)

        head = RetinaNetHead(cfg, self.fe.out_channels)
        box_coder = BoxCoder(weights=(10., 10., 5., 5.))

        box_selector_test = make_retinanet_postprocessor(cfg, box_coder, is_train=False)

        loss_evaluator = make_retinanet_loss_evaluator(cfg, box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
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
        # generate montage feature
        sizes = [f.shape[-2:] for f in features]
        features, levels = self.mf(features)

        mon_feature, montage_info = self.mb(features, sizes)

        mon_feature = self.fe(mon_feature)

        box_cls, box_regression = self.head(mon_feature, montage_info)
        anchors = self.anchor_generator(images, features)

        if self.training:
            return self._forward_train(anchors, box_cls, box_regression, targets)
        else:
            return self._forward_test(anchors, box_cls, box_regression)

    def _forward_train(self, anchors, box_cls, box_regression, targets):

        loss_box_cls, loss_box_reg = self.loss_evaluator(
            anchors, box_cls, box_regression, targets
        )
        losses = {
            "loss_retina_cls": loss_box_cls,
            "loss_retina_reg": loss_box_reg,
        }
        return anchors, losses

    def _forward_test(self, anchors, box_cls, box_regression):
        boxes = self.box_selector_test(anchors, box_cls, box_regression)
        return boxes, {}


def build_retina_pack(cfg, in_channels):
    return RetinaNetModule(cfg, in_channels)
