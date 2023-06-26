import os
import json
import shutil
from tqdm import tqdm
from pathlib import Path


# save
def save(image_paths, output_image_dir, output_label_dir):
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    annotation = []
    for path in tqdm(image_paths):
        shutil.copy(path, output_image_dir)

        # keypoints label
        keypoint_str = path.stem.split("-")[3]
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

        # save label
        annotation.append(
            {
                "image": path.name,
                "bbox": bbox,
                "keypoints": keypoints,
                "keypoints_visible": [1] * len(keypoints),
            }
        )

    with open(os.path.join(output_label_dir, "annotations.json"), "w") as f:
        json.dump(annotation, f)


if __name__ == "__main__":
    train_image_dir = "/home/cvrsg/MyData/Datasets/ALPR_Paper/ccpd_split/train"
    val_image_dir = "/home/cvrsg/MyData/Datasets/ALPR_Paper/ccpd_split/val"
    test_image_dir = "/home/cvrsg/MyData/Datasets/ALPR_Paper/ccpd_split/test"
    output_dir = "./data/ccpd"

    # parse train
    train_image_paths = list(Path(train_image_dir).glob("*.jpg"))

    save(
        train_image_paths,
        os.path.join(output_dir, "train", "images"),
        os.path.join(output_dir, "train"),
    )

    # parse val
    val_image_paths = list(Path(val_image_dir).glob("*.jpg"))

    save(
        val_image_paths,
        os.path.join(output_dir, "val", "images"),
        os.path.join(output_dir, "val"),
    )

    # parse test
    test_image_paths = list(Path(test_image_dir).glob("*.jpg"))

    save(
        test_image_paths,
        os.path.join(output_dir, "test", "images"),
        os.path.join(output_dir, "test"),
    )
