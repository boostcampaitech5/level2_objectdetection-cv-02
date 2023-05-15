from augmentation.BaseAugmentation import Make_BaseAugmentation
from augmentation.CustomAugmentation import Make_CustomAugmentation


def get_pipeline(select = "BaseAugmentation", custom_version = None, trash_norm = False) :
    """
    BaseAugmentation : default config Augmentation pipeline
    CustomAugmentation : setting Augmentation to Custom
    
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

    if select == "BaseAugmentation":     
        pipeline = Make_BaseAugmentation(img_norm_cfg)
    elif select == "CustomAugmentation":
        pipeline = Make_CustomAugmentation(img_norm_cfg)

    return pipeline

