import functools
import cv2
from collections import defaultdict
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
from .comm import is_main_process

# ref:
# * https://github.com/jytime/Mask_RCNN_Pytorch/blob/master/visualize.py
# * https://github.com/chengsq/pytorch-lighthead/blob/master/lib/model/utils/net_utils.py

DEBUG = True


def need_main_process(func):
    def wrapper(*args, **kwargs):
        if is_main_process():
            if DEBUG:
                func(*args, **kwargs)
            else:
                try:
                    func(*args, **kwargs)
                    # print("visualization: ", args, kwargs)
                except:
                    print("Meet error when perform visualization: ", args, kwargs)

    return wrapper


def to_grayscale(image):
    """
    image: tensor, chw
    """
    image = image.float().abs().mean(dim=0)
    return image


def alpha_blend(image, mask, alpha=0.5, color=(0,255,0)):
    """
    image: tensor, 3*h*w
    mask: h*w
    """
    h, w = mask.shape[-2:]
    mask = mask.view(1, h, w)

    H, W = image.shape[-2:]
    mask = F.interpolate(mask[None], size=(H, W), mode='nearest')[0]  # 1*H*W
    mask = (mask > 0.5).float()

    new_image = torch.zeros_like(image)
    for i in range(image.shape[0]):
        new_image[i, :, :] = torch.where(
            mask == 1,
            image[i, :, :],
            image[i, :, :] * (1 - alpha) + alpha * color[i]
        )

    return new_image


def has_substrings(s, subs):
    """
    s: str
    subs: list[str]
    """
    for sub in subs:
        if sub not in s:
            return False
    return True


def normalize(feature, scale=255.0):
    """
    feature: tensor
    """
    m0, m1 = feature.min(), feature.max()
    return (feature - m0) / (m1 - m0 + 1e-4) * scale


def draw_boxes(feature, boxes, color, thickness=1):
    """
    add borders to a feature map
    feature: numpy array, hwc
    box: [[x0,y0,x1,y1], ...]
    """
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        feature = cv2.rectangle(feature, (x0,y0), (x1,y1), color, thickness)
    return feature


def plot_boxes(image, boxes, scores, labels, classes=None, linestyle='solid',
               text_pos='tl', alpha=1, filename=None):
    """
    plot boxes on image
    boxes: array, xyxy
    """
    assert text_pos in ['tl', 'br']

    fig = plt.figure()
    if image is not None:
        plt.imshow(image)

    colors = plt.cm.hsv(np.linspace(0, 1, len(boxes) + 1)).tolist()
    currentAxis = plt.gca()

    for i in range(len(boxes)):
        pt = boxes[i][:4]
        label_name = classes[labels[i]] if len(labels) > 0 else 'C'
        if len(scores):
            display_txt = '%s:%s' % (label_name, round(scores[i], 4))
        else:
            display_txt = '%s' % label_name
        coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
        color = colors[i]
        currentAxis.add_patch(
            plt.Rectangle(*coords,
                          fill=False,
                          edgecolor=color,
                          linewidth=2,
                          linestyle=linestyle,
                          alpha=alpha)
        )
        if text_pos == 'tl':
            pos = pt[:2]
        else:
            pos = pt[2:]
        currentAxis.text(*pos, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

    if filename is not None:
        plt.savefig(filename)
        plt.close()

    return fig


class SummaryWriterX(SummaryWriter):
    """
    see issue: https://github.com/tensorflow/tensorflow/issues/9512
    """
    def __init__(self, logdir, env, show_period, plot_period, classes, iteration=0):
        super(SummaryWriterX, self).__init__(logdir=logdir)
        self.classes = classes
        self.vis_env = env
        self.show_period = show_period
        self.plot_period = plot_period
        self.iteration = iteration
        self.topk_pred_boxes = 100
        self.inner_curves = defaultdict(dict)
        self.batch_idx = 0  # show which image in this batch
        self.event_list = []

    @need_main_process
    def update_iteration(self, iteration):
        self.iteration = iteration

    @need_main_process
    def update_curve_values(self, group, key, value):
        if self.is_plot():
            self.inner_curves[group].update({key: value})

    def is_show(self):
        return self.iteration % self.show_period == 0

    def is_plot(self):
        return self.iteration % self.plot_period == 0

    @need_main_process
    def vis_inner_curves(self):
        """
        no period checking here
        """
        for main_tag, data in self.inner_curves.items():
            self.vis_curves(data, main_tag)

    @need_main_process
    def vis_curves(self, data_dict, main_tag):
        """
        may call this func use different periods, thus no period checking here
        """
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                data_dict[k] = v.item()  # scalar
            else:
                data_dict[k] = v
        # print(loss_dict)
        main_tag = self.vis_env + '/' + main_tag
        self.add_scalars(main_tag, data_dict, self.iteration)

    @need_main_process
    def vis_feat_hist(self, model, main_tag, filter='conv'):
        if not self.is_show():
            return

        main_tag = self.vis_env + '/' + main_tag
        with torch.no_grad():
            for i, (name, param) in enumerate(model.named_parameters()):
                if filter in name:
                    self.add_histogram(main_tag + '/' + name, param, 0)

    @need_main_process
    def vis_image(self, image, main_tag, targets=None):
        """
        show boxes on an image
        image: c*h*w
        boxes: BoxList
        """
        if not self.is_show():
            return

        main_tag = self.vis_env + '/' + main_tag
        image = image.cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)  # HWC
        if targets is None:
            fig = plt.figure()
            plt.imshow(image)
        else:
            boxes_ = targets.bbox.cpu().data.numpy().astype(int)
            if targets.has_field('labels'):
                labels_ = targets.get_field('labels').cpu().data.numpy()
            else:
                labels_ = []
            if targets.has_field('scores'):
                scores_ = targets.get_field('scores').cpu().data.numpy()
            else:
                scores_ = []

            if len(scores_):
                id_sort = np.argsort(scores_)[::-1]
                id_sort = id_sort[:self.topk_pred_boxes].tolist()
                scores_ = scores_[id_sort]
                if len(labels_):
                    labels_ = labels_[id_sort]
                boxes_ = boxes_[id_sort]

            boxes_ = boxes_.tolist()
            fig = plot_boxes(
                image,  # need HWC
                boxes_, scores_,
                labels_,
                classes=self.classes,
                linestyle='solid',
                text_pos='tl',
                alpha=1
            )
        self.add_figure(main_tag, fig, self.iteration)

    @need_main_process
    def vis_image_points(self, image, points, main_tag, color='r'):
        """
        show an image with some points plotted on it
        points: n*2, xy
        """
        if not self.is_show():
            return

        main_tag = self.vis_env + '/' + main_tag
        image = image.cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)  # HWC
        points = points.cpu().data.numpy()
        fig = plt.figure()
        plt.imshow(image)
        if len(points):
            plt.scatter(x=points[:,0], y=points[:,1], c=color, s=40, alpha=0.5)
        self.add_figure(main_tag, fig, self.iteration)

    @need_main_process
    def vis_feature_map(self, feature_map, main_tag):
        """
        visualize a feature map
        feature_map: c*h*w
        """
        if not self.is_show():
            return

        main_tag = self.vis_env + '/' + main_tag
        feature_map = normalize(to_grayscale(feature_map.data)).cpu().numpy().astype(np.uint8)  # hw
        feature_map = cv2.applyColorMap(feature_map, cv2.COLORMAP_JET)[:, :, ::-1]  # hwc

        fig = plt.figure()
        plt.imshow(feature_map)
        self.add_figure(main_tag, fig, self.iteration)
        # print(self.iteration, main_tag)

    @need_main_process
    def vis_mon_feature_map(self, feature_map, ranges, main_tag):
        """
        visualize a montage feature map
        feature_map: c*h*w
        """
        if not self.is_show():
            return

        if ranges is None:
            self.vis_feature_map(feature_map, main_tag)
            return

        main_tag = self.vis_env + '/' + main_tag
        # print(feature_map.shape)
        feature_map = to_grayscale(feature_map.data).cpu().data
        if isinstance(ranges, torch.Tensor):
            ranges = ranges.cpu().data.numpy()
        for i, rg in enumerate(ranges):
            x0, y0, x1, y1 = rg
            feature_map[y0:y1, x0:x1] = normalize(feature_map[y0:y1, x0:x1])
        feature_map = feature_map.numpy().astype(np.uint8)
        feature_map = cv2.applyColorMap(feature_map, cv2.COLORMAP_JET)  # hwc
        feature_map = draw_boxes(feature_map, ranges, (255, 255, 255), thickness=1)
        feature_map = feature_map[:, :, ::-1]  # to RGB

        fig = plt.figure()
        plt.imshow(feature_map)
        self.add_figure(main_tag, fig, self.iteration)
        # print(self.iteration, main_tag)

    @need_main_process
    def add_event(self, event, *args, **kwargs):
        if not self.is_show():
            return
        self.event_list.append(functools.partial(event, *args, **kwargs))

    @need_main_process
    def digest_events(self):
        for event in self.event_list:
            event()
        self.event_list = []
