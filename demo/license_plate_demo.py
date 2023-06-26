# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

from mmcv.image import imread

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--img_dir",
        default="/home/cvrsg/MyData/Datasets/ALPR_Paper/ccpd_split/test",
        help="Image dir",
    )
    parser.add_argument(
        "--config",
        default="configs/license_plate_2d_keypoint/rtmpose/tjlp/rtmpose-t_420e_tjlp-160x160.py",
        # default="configs/license_plate_2d_keypoint/rtmpose/ccpd/rtmpose-t_420e_ccpd-160x160.py",
        help="Config file",
    )
    parser.add_argument(
        "--checkpoint",
        default="work_dirs/rtmpose-t_420e_tjlp-160x160/best_0.02_PCK_epoch_407.pth",
        # default="work_dirs/rtmpose-t_420e_ccpd-160x160/best_0.02_PCK_epoch_417.pth",
        help="Checkpoint file",
    )
    parser.add_argument("--dataset-type", default="ccpd", help="Dataset type")
    parser.add_argument("--out-file", default="test.jpg", help="Path to output file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--draw-heatmap", action="store_true", default=False, help="Visualize the predicted heatmap"
    )
    parser.add_argument(
        "--show-kpt-idx",
        action="store_true",
        default=False,
        help="Whether to show the index of keypoints",
    )
    parser.add_argument(
        "--skeleton-style",
        default="mmpose",
        type=str,
        choices=["mmpose", "openpose"],
        help="Skeleton style selection",
    )
    parser.add_argument(
        "--kpt-thr", type=float, default=0.3, help="Visualizing keypoint thresholds"
    )
    parser.add_argument(
        "--radius", type=int, default=3, help="Keypoint radius for visualization"
    )
    parser.add_argument(
        "--thickness", type=int, default=1, help="Link thickness for visualization"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.8, help="The transparency of bboxes"
    )
    parser.add_argument(
        "--show", action="store_true", default=False, help="whether to show img"
    )
    args = parser.parse_args()
    return args


def ccpd_bounding_box(image_path):
    """Returns the bounding box of a ccpd image."""
    keypoint_str = image_path.stem.split("-")[3]
    keypoint_str = keypoint_str.split("_")
    keypoints = [s.split("&") for s in keypoint_str]
    keypoints = [[int(s[0]), int(s[1])] for s in keypoints]

    # bbox (xmin, ymin, xmax, ymax)
    bbox = [
        min([kp[0] for kp in keypoints]),
        min([kp[1] for kp in keypoints]),
        max([kp[0] for kp in keypoints]),
        max([kp[1] for kp in keypoints]),
    ]

    return bbox


def ccpd_keypoints(image_path):
    """Returns the keypoints of a ccpd image."""
    keypoint_str = image_path.stem.split("-")[3]
    keypoint_str = keypoint_str.split("_")
    keypoints = [s.split("&") for s in keypoint_str]
    keypoints = [[int(s[0]), int(s[1])] for s in keypoints]

    return keypoints


def tjlp_bounding_box(image_path):
    """Returns the bounding box of a tjlp image."""
    keypoints = image_path.stem.split("_")[1].split("-")
    keypoints = [[int(x) for x in kp.split("&")] for kp in keypoints]

    # bbox (xmin, ymin, xmax, ymax)
    bbox = [
        min([kp[0] for kp in keypoints]),
        min([kp[1] for kp in keypoints]),
        max([kp[0] for kp in keypoints]),
        max([kp[1] for kp in keypoints]),
    ]

    return bbox


def tjlp_keypoints(image_path):
    """Returns the keypoints of a tjlp image."""
    keypoints = image_path.stem.split("_")[1].split("-")
    keypoints = [[int(x) for x in kp.split("&")] for kp in keypoints]

    return keypoints


def main():
    args = parse_args()

    # build the model from a config file and a checkpoint file
    if args.draw_heatmap:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    else:
        cfg_options = None

    model = init_model(
        args.config, args.checkpoint, device=args.device, cfg_options=cfg_options
    )

    # init visualizer
    model.cfg.visualizer.radius = args.radius
    model.cfg.visualizer.alpha = args.alpha
    model.cfg.visualizer.line_width = args.thickness

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(model.dataset_meta, skeleton_style=args.skeleton_style)

    image_paths = list(Path(args.img_dir).glob("*.jpg"))

    for image_path in image_paths:
        if args.dataset_type == "ccpd":
            bbox = [ccpd_bounding_box(image_path)]
            batch_results = inference_topdown(
                model, str(image_path), bbox, bbox_format="xyxy"
            )
        elif args.dataset_type == "tjlp":
            bbox = [tjlp_bounding_box(image_path)]
            batch_results = inference_topdown(
                model, str(image_path), bbox, bbox_format="xyxy"
            )
        else:
            batch_results = inference_topdown(model, str(image_path))

        results = merge_data_samples(batch_results)

        # show the results
        img = imread(str(image_path), channel_order="rgb")

        visualizer.add_datasample(
            "result",
            img,
            data_sample=results,
            draw_gt=False,
            draw_bbox=False,
            kpt_thr=args.kpt_thr,
            draw_heatmap=args.draw_heatmap,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            out_file=args.out_file,
        )

        if args.dataset_type == "ccpd":
            # Draw the keypoints
            import cv2

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            keypoints = ccpd_keypoints(image_path)
            for i, kp in enumerate(keypoints):
                cv2.circle(img, (kp[0], kp[1]), 2, (0, 0, 255), 2)
                cv2.putText(
                    img,
                    str(i),
                    (kp[0], kp[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

            cv2.imwrite("test_2.jpg", img)
        elif args.dataset_type == "tjlp":
            # Draw the keypoints
            import cv2

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            keypoints = tjlp_keypoints(image_path)
            for i, kp in enumerate(keypoints):
                cv2.circle(img, (kp[0], kp[1]), 2, (0, 0, 255), 2)
                cv2.putText(
                    img,
                    str(i),
                    (kp[0], kp[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

            cv2.imwrite("test_2.jpg", img)
        else:
            pass

        print(results.pred_instances.keypoint_scores[0])


if __name__ == "__main__":
    main()
