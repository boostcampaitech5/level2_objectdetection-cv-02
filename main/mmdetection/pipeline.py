


def get_trainpipeline(select = "BaseAugmentation", trash_norm = False) :
    """
    BaseAugmentation : default config Augmentation pipeline
    
    """
    if trash_norm : 
        img_norm_cfg = dict(  #"Trash dataset mean and std"
            mean=[0.43179792, 0.4604544,  0.48490757],     
            std=[0.2147372,  0.20920224, 0.21176134], 
            to_rgb=True)
    else : 
        img_norm_cfg = dict( #"COCO dataset mean and std"
            mean=[123.675, 116.28, 103.53],     
            std=[58.395, 57.12, 57.375], 
            to_rgb=True)
         
    if select == "BaseAugmentation" :
        pipeline = [      
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]
    return pipeline



def get_valpipeline(select = "BaseAugmentation", trash_norm = False) :
    if trash_norm : 
        img_norm_cfg = dict(  #"Trash dataset mean and std"
            mean=[0.43179792, 0.4604544,  0.48490757],     
            std=[0.2147372,  0.20920224, 0.21176134], 
            to_rgb=True)
    else : 
        img_norm_cfg = dict( #"COCO dataset mean and std"
            mean=[123.675, 116.28, 103.53],     
            std=[58.395, 57.12, 57.375], 
            to_rgb=True)
         
    if select == "BaseAugmentation" :
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
        ])
]
    return pipeline

def get_testpipeline(select = "BaseAugmentation", trash_norm = False) : 
    if trash_norm : 
        img_norm_cfg = dict(  #"Trash dataset mean and std"
            mean=[0.43179792, 0.4604544,  0.48490757],     
            std=[0.2147372,  0.20920224, 0.21176134], 
            to_rgb=True)
    else : 
        img_norm_cfg = dict( #"COCO dataset mean and std"
            mean=[123.675, 116.28, 103.53],     
            std=[58.395, 57.12, 57.375], 
            to_rgb=True)
         
    if select == "BaseAugmentation" :
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
                ])
        ]
    return pipeline