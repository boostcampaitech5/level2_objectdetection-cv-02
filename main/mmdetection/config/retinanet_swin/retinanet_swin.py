_base_ = [ '/opt/ml/baseline/mmdetection/configs/_base_/models/retinanet_r50_swimL.py', '/opt/ml/baseline/UniverseNet/configs/_base_/datasets/coco_detection_mstrain_480_960_TTA.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# optimizer

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(warmup_iters=1000)
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
#fp16 = dict(loss_scale=512.)
