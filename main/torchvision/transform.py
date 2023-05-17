import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(aug_cfgs):
    """train dataset에 가할 augmentation을 정의하는 함수.

    Args:
        aug_cfgs (dict): augmentation 관련 설정 dictionary.

    Returns:
        _type_: augmentation container
    """
    return A.Compose([
        A.Resize(aug_cfgs['img_height'], aug_cfgs['img_width']),
        A.RandomBrightness(p=0.2),
        A.RandomContrast(p=0.2),
        A.Flip(p=0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_valid_transform(aug_cfgs):
    """validation dataset에 가할 augmentation을 정의하는 함수.

    Args:
        aug_cfgs (dict): augmentation 관련 설정 dictionary.

    Returns:
        _type_: augmentation container
    """
    return A.Compose([
        A.Resize(aug_cfgs['img_height'], aug_cfgs['img_width']),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
