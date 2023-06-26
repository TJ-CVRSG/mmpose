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
        keypoints = path.stem.split("_")[1].split("-")
        keypoints = [[int(x) for x in kp.split("&")] for kp in keypoints]

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
    train_image_dir = "/home/cvrsg/MyData/TJLP_Dataset/train"
    test_image_dir = "/home/cvrsg/MyData/TJLP_Dataset/test"
    output_dir = "./data/tjlp"

    # parse train
    train_image_paths = list(Path(train_image_dir).glob("*.jpg"))

    save(
        train_image_paths,
        os.path.join(output_dir, "train", "images"),
        os.path.join(output_dir, "train"),
    )

    # parse test
    test_image_paths = list(Path(test_image_dir).glob("*.jpg"))

    save(
        test_image_paths,
        os.path.join(output_dir, "test", "images"),
        os.path.join(output_dir, "test"),
    )
