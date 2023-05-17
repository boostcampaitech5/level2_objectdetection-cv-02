import torchvision

from torch.hub import load_state_dict_from_url
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN


def get_fasterrcnn_resnet50_fpn(pretrained:bool=True, progress:bool=True,
                                pretrained_backbone:bool=True, num_classes:int=11,
                                trainable_backbone_layers:int=3):
    

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained,
                                                                 progress=progress,
                                                                 pretrained_backbone=pretrained_backbone,
                                                                 trainable_backbone_layers=trainable_backbone_layers)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)                                

    return model
