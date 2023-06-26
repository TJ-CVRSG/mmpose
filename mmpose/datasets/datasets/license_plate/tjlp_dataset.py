# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine.fileio import exists, get_local_path
from scipy.io import loadmat

from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class TJLPDataset(BaseCocoStyleDataset):
    """TJLP dataset for pose estimation.


    TJLP keypoints::

        0: 'bottom_right',
        1: 'bottom_left',
        2: 'top_left',
        3: 'top_right',

    Args:
        ann_file (str): Annotation file path. Default: ''.
        bbox_file (str, optional): Detection result file path. If
            ``bbox_file`` is set, detected bboxes loaded from this file will
            be used instead of ground-truth bboxes. This setting is only for
            evaluation, i.e., ignored when ``test_mode`` is ``False``.
            Default: ``None``.
        data_mode (str): Specifies the mode of data samples: ``'topdown'`` or
            ``'bottomup'``. In ``'topdown'`` mode, each data sample contains
            one instance; while in ``'bottomup'`` mode, each data sample
            contains all instances in a image. Default: ``'topdown'``
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Default: ``None``.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Default: ``None``.
        data_prefix (dict, optional): Prefix for training data. Default:
            ``dict(img=None, ann=None)``.
        filter_cfg (dict, optional): Config for filter data. Default: `None`.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Default: ``None`` which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy.
            Default: ``True``.
        pipeline (list, optional): Processing pipeline. Default: [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Default: ``False``.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Default: ``False``.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Default: 1000.
    """

    METAINFO: dict = dict(from_file="configs/_base_/datasets/tjlp.py")

    def __init__(
        self,
        ann_file: str = "",
        bbox_file: Optional[str] = None,
        headbox_file: Optional[str] = None,
        data_mode: str = "topdown",
        metainfo: Optional[dict] = None,
        data_root: Optional[str] = None,
        data_prefix: dict = dict(img=""),
        filter_cfg: Optional[dict] = None,
        indices: Optional[Union[int, Sequence[int]]] = None,
        serialize_data: bool = True,
        pipeline: List[Union[dict, Callable]] = [],
        test_mode: bool = False,
        lazy_init: bool = False,
        max_refetch: int = 1000,
    ):
        self.headbox_file = headbox_file

        super().__init__(
            ann_file=ann_file,
            bbox_file=bbox_file,
            data_mode=data_mode,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
        )

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        """Load data from annotations in CVRSG format."""

        assert exists(self.ann_file), "Annotation file does not exist"
        with get_local_path(self.ann_file) as local_path:
            with open(local_path) as anno_file:
                self.anns = json.load(anno_file)

        if self.headbox_file:
            assert exists(self.headbox_file), "Headbox file does not exist"
            with get_local_path(self.headbox_file) as local_path:
                self.headbox_dict = loadmat(local_path)
            headboxes_src = np.transpose(self.headbox_dict["headboxes_src"], [2, 0, 1])
            SC_BIAS = 0.6

        instance_list = []
        image_list = []
        used_img_ids = set()
        ann_id = 0

        for idx, ann in enumerate(self.anns):
            # load keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
            keypoints = np.array(ann["keypoints"]).reshape(1, -1, 2)
            keypoints_visible = np.array(ann["keypoints_visible"]).reshape(1, -1)

            # load bbox in shape [1, 4]
            bbox = np.array(ann["bbox"]).reshape(1, -1)

            instance_info = {
                "id": ann_id,
                "img_id": ann["image"].split(".")[0],
                "img_path": osp.join(self.data_prefix["img"], ann["image"]),
                "bbox": bbox,
                "bbox_score": np.ones(1, dtype=np.float32),
                "keypoints": keypoints,
                "keypoints_visible": keypoints_visible,
            }

            if self.headbox_file:
                # calculate the diagonal length of head box as norm_factor
                headbox = headboxes_src[idx]
                head_size = np.linalg.norm(headbox[1] - headbox[0], axis=0)
                head_size *= SC_BIAS
                instance_info["head_size"] = head_size.reshape(1, -1)

            if instance_info["img_id"] not in used_img_ids:
                used_img_ids.add(instance_info["img_id"])
                image_list.append(
                    {
                        "img_id": instance_info["img_id"],
                        "img_path": instance_info["img_path"],
                    }
                )

            instance_list.append(instance_info)
            ann_id = ann_id + 1

        return instance_list, image_list
