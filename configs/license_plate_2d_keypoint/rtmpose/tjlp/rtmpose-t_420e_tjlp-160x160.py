_base_ = ["../../../_base_/default_runtime.py"]

# runtime
max_epochs = 420
stage2_num_epochs = 30
base_lr = 4e-3

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.0),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

# learning rate
param_scheduler = [
    dict(type="LinearLR", start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(
        # use cosine lr from 210 to 420 epoch
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=1024)

# codec settings
codec = dict(
    type="SimCCLabel",
    input_size=(160, 160),
    sigma=(4.47, 4.47),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)

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
        _scope_="mmdet",
        type="CSPNeXt",
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.375,
        out_indices=(4,),
        channel_attention=True,
        norm_cfg=dict(type="SyncBN"),
        act_cfg=dict(type="SiLU"),
    ),
    head=dict(
        type="RTMCCHead",
        in_channels=384,
        out_channels=4,
        input_size=codec["input_size"],
        in_featuremap_size=(5, 5),
        simcc_split_ratio=codec["simcc_split_ratio"],
        final_layer_kernel_size=3,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn="SiLU",
            use_rel_bias=False,
            pos_enc=False,
        ),
        loss=dict(
            type="KLDiscretLoss", use_target_weight=True, beta=10.0, label_softmax=True
        ),
        decoder=codec,
    ),
    test_cfg=dict(flip_test=False),
)

# base dataset settings
dataset_type = "TJLPDataset"
data_mode = "topdown"
data_root = "data/tjlp/"

backend_args = dict(backend="local")

# pipelines
train_pipeline = [
    dict(type="LoadImage", backend_args=backend_args),
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
    dict(type="LoadImage", backend_args=backend_args),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="PackPoseInputs"),
]

train_pipeline_stage2 = [
    dict(type="LoadImage", backend_args=backend_args),
    dict(type="GetBBoxCenterScale"),
    dict(
        type="RandomBBoxTransform",
        shift_factor=0.0,
        scale_factor=[1, 1.2],
        rotate_factor=30,
    ),
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
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5,
            ),
        ],
    ),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

# data loaders
train_dataloader = dict(
    batch_size=256,
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
    batch_size=256,
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

custom_hooks = [
    dict(
        type="mmdet.PipelineSwitchHook",
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2,
    )
]

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
            project="plate_loc_paper", name="rtmpose-tiny-tjlp", entity="ytang"
        ),
    ),
]
visualizer = dict(
    type="PoseLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
