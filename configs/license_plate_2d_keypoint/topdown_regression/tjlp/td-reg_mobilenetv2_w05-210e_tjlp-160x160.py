_base_ = ["../../../_base_/default_runtime.py"]

# runtime
train_cfg = dict(max_epochs=210, val_interval=1)

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type="Adam",
        lr=1e-3,
    )
)

# learning policy
param_scheduler = [
    dict(
        type="LinearLR", begin=0, end=500, start_factor=0.001, by_epoch=False
    ),  # warm-upd
    dict(
        type="MultiStepLR",
        begin=0,
        end=train_cfg["max_epochs"],
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True,
    ),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# codec settings
codec = dict(type="RegressionLabel", input_size=(160, 160))

# model settings
model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type="MobileNetV2",
        widen_factor=0.5,
        out_indices=(7,),
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="RegressionHead",
        in_channels=1280,
        num_joints=4,
        loss=dict(type="SmoothL1Loss", use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=False,
        shift_coords=True,
    ),
)

# base dataset settings
dataset_type = "TJLPDataset"
data_mode = "topdown"
data_root = "data/tjlp/"

# pipelines
train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomBBoxTransform", scale_factor=[0.9, 1.3], rotate_factor=30),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="mmdet.YOLOXHSVRandomAug"),
    dict(
        type="Albumentation",
        transforms=[
            dict(type="Blur", p=0.1),
            dict(type="MedianBlur", p=0.1),
            dict(
                type="CoarseDropout",
                max_holes=1,
                max_height=0.3,
                max_width=0.3,
                min_holes=1,
                min_height=0.15,
                min_width=0.15,
                p=1.0,
            ),
        ],
    ),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]
val_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="PackPoseInputs"),
]

# data loaders
train_dataloader = dict(
    batch_size=32,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="train/annotations.json",
        data_prefix=dict(img="train/images/"),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=64,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="test/annotations.json",
        data_prefix=dict(img="test/images/"),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(save_best="0.02/PCK", rule="greater", max_keep_ckpts=1)
)

# evaluators
val_evaluator = [
    dict(type="PCKAccuracy", thr=0.02, prefix="0.02"),
    dict(type="PCKAccuracy", thr=0.05, prefix="0.05"),
]
test_evaluator = val_evaluator

# visualizer
vis_backends = [
    dict(type="LocalVisBackend"),
    dict(
        type="WandbVisBackend",
        init_kwargs=dict(
            project="plate_loc_paper", name="mobilenetv2_a05-l1-tjlp-160x160", entity="tj_cvrsg"
        ),
    ),
]
visualizer = dict(
    type="PoseLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
