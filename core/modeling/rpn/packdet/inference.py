import torch

from ..inference import RPNPostProcessor
from ..utils import permute_and_flatten

from core.modeling.box_coder import BoxCoder
from core.modeling.utils import cat
from core.structures.bounding_box import BoxList
from core.structures.boxlist_ops import cat_boxlist
from core.structures.boxlist_ops import boxlist_ml_nms
from core.structures.boxlist_ops import remove_small_boxes


class PACKDETPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        bbox_aug_enabled=False
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(PACKDETPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.bbox_aug_enabled = bbox_aug_enabled

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness,
            image_sizes):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            detections = torch.cat([
                per_locations[:, :2] - per_box_regression[:, :2],
                per_locations[:, :2] + per_box_regression[:, 2:4],
            ], dim=1)

            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", torch.sqrt(per_box_cls))
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def merge_same_scales(self, box_cls, box_regression, centerness, montage_info):
        levels = torch.tensor(montage_info['levels']).to(box_cls[0].device)
        sizes = [x.shape[-2:] for x in box_cls]
        locations_map = montage_info['locations_map']

        locations = [loc.view(siz[0], siz[1], -1) for loc, siz in zip(locations_map, sizes)]
        new_levels, cnts = torch.unique(torch.tensor(levels).int(), sorted=True, return_counts=True)

        if len(new_levels) == len(levels):
            return box_cls, box_regression, centerness, montage_info['locations_map'], levels

        new_box_cls = []
        new_box_regression = []
        new_centerness = []
        new_locations = []
        for l in new_levels:
            indices = torch.nonzero(levels == l.item()).view(-1)
            cur_box_cls = torch.cat([box_cls[i] for i in indices], dim=2)  # horizontally
            cur_box_regression = torch.cat([box_regression[i] for i in indices], dim=2)
            cur_centerness = torch.cat([centerness[i] for i in indices], dim=2)
            cur_locations = torch.cat([locations[i] for i in indices], dim=0)
            cur_locations = cur_locations.view(-1, 2)
            new_box_cls.append(cur_box_cls)
            new_box_regression.append(cur_box_regression)
            new_centerness.append(cur_centerness)
            new_locations.append(cur_locations)

        return new_box_cls, new_box_regression, new_centerness, new_locations, new_levels

    def forward(self, montage_info, box_cls, box_regression, centerness, image_sizes):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        box_cls, box_regression, centerness, locations, levels = self.merge_same_scales(
            box_cls, box_regression, centerness, montage_info
        )

        sampled_boxes = []
        for _, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, b, c, image_sizes
                )
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        if not self.bbox_aug_enabled:
            boxlists = self.select_over_all_levels(boxlists, box_cls[0].device)

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists, device):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = boxlist_ml_nms(boxlists[i], self.num_classes, self.nms_thresh, device=device)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


def make_packdet_postprocessor(config):
    pre_nms_thresh = config.MODEL.PACKDET.INFERENCE_TH
    pre_nms_top_n = config.MODEL.PACKDET.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.PACKDET.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    bbox_aug_enabled = config.TEST.BBOX_AUG.ENABLED

    box_selector = PACKDETPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config.MODEL.PACKDET.NUM_CLASSES-1,
        bbox_aug_enabled=bbox_aug_enabled
    )

    return box_selector
