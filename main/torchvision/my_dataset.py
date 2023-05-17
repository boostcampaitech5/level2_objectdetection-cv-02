import torch
from torch.utils.data import Dataset

import numpy as np
from pycocotools.coco import COCO

import random
import cv2
import os
import math


class CustomDataset(Dataset):
    def __init__(self, annotation, data_dir, transforms=None):
        """object detection 대회를 위한 custom dataset

        Args:
            annotation (_type_): object detection dataset의 annotation 경로
            data_dir (_type_): annoation으로부터 이미지를 불러오기 위한 dataset 경로
            transforms (_type_, optional): 이미지에 가할 transformation. Defaults to None.
        """
        super(CustomDataset, self).__init__()
        self.data_dir = data_dir

        # coco annotation 불러오기 (coco API)
        self.coco = COCO(annotation)
        self.image_ids = self.coco.getImgIds() # type: list

        self.transforms = transforms
    

    def __getitem__(self, index: int):
        image_id = self.coco.getImgIds(imgIds=index)

        try:
            image_info = self.coco.loadImgs(image_id)[0]
        except KeyError: # image_ids에 없는 image_id를 조회할 경우, image_ids에서 무작위로 하나를 추출해서 batch를 구성
            image_id = random.choice(self.image_ids)
        
        image_info = self.coco.loadImgs(image_id)[0]
            
        image = cv2.imread(os.path.join(self.data_dir, image_info["file_name"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        anns = self.coco.loadAnns(ann_ids)

        boxes = np.array([x['bbox'] for x in anns])

        # (x_min, y_min, width, height) -> (x_min, y_min, x_max, y_max)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        # torchvision faster_rcnn은 label=0을 background로 취급
        # class_id를 1~10으로 수정
        labels = np.array([x['category_id']+1 for x in anns])
        labels = torch.as_tensor(labels, dtype=torch.int64)

        areas = np.array([x['area'] for x in anns])
        areas = torch.as_tensor(areas, dtype=torch.float32)

        is_crowds = np.array([x['iscrowd'] for x in anns])
        is_crowds = torch.as_tensor(is_crowds, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels,
                'image_id': torch.tensor([image_id]), 'area': areas,
                'iscrowd': is_crowds}
        
        # transform
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            target['boxes'] = torch.tensor(sample['bboxes'], dtype=torch.float32)

        return image, target, torch.tensor([image_id])

    
    def __len__(self):
        return len(self.coco.getImgIds())


class TestDataset(Dataset):
    def __init__(self, annotation, data_dir):
        """Test를 위한 데이터셋

        Args:
            annotation (_type_): test data에 대한 정보를 가지고 있는 json 파일 경로
            data_dir (_type_): 원본 이미지 데이터의 경로
        """
        super().__init__()
        self.data_dir = data_dir
        # coco annotation 불러오기 (coco API)
        self.coco = COCO(annotation)

    def __getitem__(self, index: int):
        
        image_id = self.coco.getImgIds(imgIds=index)

        image_info = self.coco.loadImgs(image_id)[0]
        
        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        anns = self.coco.loadAnns(ann_ids)

        image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)

        return image
    
    def __len__(self) -> int:
        return len(self.coco.getImgIds())


# Test Code
if __name__ == "__main__":
    annotation = "/opt/ml/baseline/level2_objectdetection-cv-02/main/detectron2/json/detectron2_train.json"
    data_dir = "/opt/ml/dataset"

