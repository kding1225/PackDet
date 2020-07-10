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
            image_sizes, scales_map):
        """
        Arguments:
            locations: N*2
            box_cls: N*C*H*W
            box_regression: N*4*H*W
            centerness: N*1*H*W
        """
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()  # N*(HW)*C
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)  # N*(HW)*4
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()  # N*(HW)*1
        scales_map = scales_map.view(-1)

        candidate_inds = box_cls > self.pre_nms_thresh

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]  # (HW)*C
            per_candidate_inds = candidate_inds[i]  # (HW)*C
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]  # linear indices of these cands in cur image
            per_class = per_candidate_nonzeros[:, 1] + 1  # class label of these cands

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]  # prediected pos offsets of these cands
            per_locations = locations[per_box_loc]  # sampling pos of these cands
            per_scales_map = scales_map[per_box_loc]

            # loop over scales
            if len(per_scales_map) > self.pre_nms_top_n:
                idx_per_scales = []
                for j in range(5):
                    idx_per_scale = torch.nonzero(per_scales_map == j).view(-1)
                    if len(idx_per_scale) > self.pre_nms_top_n:
                        _, idx = per_box_cls[idx_per_scale].topk(self.pre_nms_top_n, sorted=False)
                        idx_per_scale = idx_per_scale[idx]
                    idx_per_scales.append(idx_per_scale)
                top_k_indices = torch.cat(idx_per_scales)
                per_box_cls = per_box_cls[top_k_indices]
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
        locations = montage_info["locations_map"]
        scales_map = montage_info["scales_map"]
        locations = locations.view(-1, 2)
        boxlists = self.forward_for_single_feature_map(
            locations, box_cls, box_regression, centerness,
            image_sizes, scales_map
        )

        if not self.bbox_aug_enabled:
            boxlists = self.select_over_all_levels(boxlists, box_cls.device)

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


def make_packdet_postprocessor2(config):
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
