# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2, os

from core.config import cfg
from predictor import COCODemo

import time


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="configs/packdet/packdet_R_50_FPN_1x_fe-128-12-2_m4_sep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weights",
        default="models/packdet_R_50_FPN_1x_fe-128-12-2_m4_sep.pth",
        metavar="FILE",
        help="path to the trained model",
    )
    parser.add_argument(
        "--images-dir",
        default="demo/images",
        metavar="DIR",
        help="path to demo images directory",
    )
    parser.add_argument(
        "--results-dir",
        default="demo/results",
        metavar="DIR",
        help="path to save the results",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHT = args.weights

    cfg.freeze()

    # The following per-class thresholds are computed by maximizing
    # per-class f-measure in their precision-recall curve.
    # Please see compute_thresholds_for_classes() in coco_eval.py for details.
    # you could copy the thrs in packdet/packdet.py here
    thresholds_for_classes = [0.4902425706386566, 0.5381519794464111, 0.5067052841186523, 0.5437142252922058, 0.5588839054107666, 0.5276558995246887, 0.49406325817108154, 0.49073269963264465, 0.4806738495826721, 0.4823538064956665, 0.6076132655143738, 0.6440929770469666, 0.5771225690841675, 0.5104134678840637, 0.518393337726593, 0.5853402018547058, 0.5871560573577881, 0.5754503607749939, 0.5476232767105103, 0.5239601135253906, 0.528354823589325, 0.5842380523681641, 0.529585063457489, 0.5392818450927734, 0.4744971990585327, 0.5273094773292542, 0.47029709815979004, 0.47505930066108704, 0.4859939515590668, 0.5534765124320984, 0.44751793146133423, 0.586391031742096, 0.5289603471755981, 0.4418090879917145, 0.49789053201675415, 0.5277994871139526, 0.5256999731063843, 0.49595320224761963, 0.4759668707847595, 0.5057507753372192, 0.47086426615715027, 0.5383470058441162, 0.5014472603797913, 0.4778696298599243, 0.4438300132751465, 0.5047871470451355, 0.4818137586116791, 0.4980989098548889, 0.5201641917228699, 0.493553102016449, 0.4972984790802002, 0.49542945623397827, 0.5166008472442627, 0.5381780862808228, 0.45813074707984924, 0.4879375994205475, 0.4892300069332123, 0.525837779045105, 0.4814700484275818, 0.4686848223209381, 0.4951763153076172, 0.5807622075080872, 0.572121262550354, 0.5322854518890381, 0.4996817708015442, 0.47158145904541016, 0.5802334547042847, 0.5267660617828369, 0.5192787647247314, 0.5104144811630249, 0.492986798286438, 0.5321716070175171, 0.5432604551315308, 0.43284663558006287, 0.572494626045227, 0.4790349006652832, 0.5585607886314392, 0.5282813310623169, 0.3120556175708771, 0.5559245347976685]

    demo_im_names = os.listdir(args.images_dir)
    demo_im_names.sort()
    print('{} images to test'.format(len(demo_im_names)))

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_thresholds_for_classes=thresholds_for_classes,
        min_image_size=args.min_image_size
    )

    # for im_name in demo_im_names:
    #     img = cv2.imread(os.path.join(args.images_dir, im_name))
    #     if img is None:
    #         continue
    #     start_time = time.time()
    #     composite = coco_demo.run_on_opencv_image(img)
    #     print("{}\tinference time: {:.2f}s".format(im_name, time.time() - start_time))
    #     cv2.imwrite(os.path.join('result', im_name), composite)
    #     # cv2.imshow(im_name, composite)
    # print("Press any keys to exit ...")
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # plt
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    for i, im_name in enumerate(demo_im_names):
        img = cv2.imread(os.path.join(args.images_dir, im_name))
        if img is None:
            continue
        start_time = time.time()
        coco_demo.run_det_on_opencv_image_plt(img, os.path.join(args.results_dir, im_name))
        print("{}, {}\tinference time: {:.2f}s".format(i, im_name, time.time() - start_time))
    print("Done!")


if __name__ == "__main__":
    main()

