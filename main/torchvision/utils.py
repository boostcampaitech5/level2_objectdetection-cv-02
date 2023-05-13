import torch

from map_boxes import mean_average_precision_for_boxes
from pycocotools.coco import COCO
import numpy as np
import pandas as pd

from tqdm import tqdm
import yaml
import random
import datetime


class Averager:
    """
    training 과정의 loss를 계산할 때 사용하는 클래스입니다.
    """
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
        
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def get_save_folder_name():
    """파일을 날짜별로 폴더에 저장하고 싶을 때, 폴더명을 만들어주는 함수

    Returns:
       str: "2023-05-12"와 같이 폴더명으로 사용할 문자열 반환
    """
    today = datetime.datetime.now()
    save_folder_name = f"{today.year}-{today.month}-{today.day}"

    return save_folder_name


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def seed_everything(seed: int):
    """실험 재현을 위해 seed를 설정하는 함수

    Args:
        seed (int): seed 값
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def load_config(config_file):
    """정의한 YAML 파일을 불러오는 함수

    Args:
        config_file : 실험에 필요한 설정들을 저장해둔 yaml 파일
    """
    with open(config_file) as file:
        config = yaml.safe_load(file)

    return config


def collate_fn(batch):
    return tuple(zip(*batch))


# TODO: 코드 정리 필요
def calculate_mAP(preds, gt_anns_path, score_threshold):
    """제공받았던 mission 코드를 변형하여, mAP를 계산해주는 함수.

    Args:
        preds (_type_): 모델이 예측한 결과로, list[dict] 형태를 따름.
                        하나의 dict는 image_id, boxes, scores, labels에 대한 정보를 담고 있음.
        gt_anns_path (_type_): 모델이 예측한 결과와 비교할 정답 파일이 존재하는 경로.
        score_threshold (_type_): mAP를 계산할 때 기준이 될 IoU threshold 값.
                                낮을수록 mAP는 높아지고, 높을수록 mAP는 낮아짐.

    Returns:
        _type_: 계산한 mAP 값과 클래스 별 AP 값을 반환함.
    """
    coco = COCO(gt_anns_path)

    # outputs을 submission 파일을 생성하는 것처럼 형식에 맞춰 변환하기
    bboxes = []
    file_names = []

    imgIDs = []
    for pred in preds:
        imgID = pred['image_id'][0]
        imgIDs.append(imgID)
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=imgID))[0]
        for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
            if score > score_threshold:
                prediction_string += str(label-1) + ' ' + str(score) + ' ' + str(box[0]) + ' ' + str(
                    box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '                
        bboxes.append(prediction_string)
        file_names.append(image_info['file_name'])
    
    new_pred = []
    # check variable type
    for i, bbox in enumerate(bboxes):
        if isinstance(bbox, float):
            print(f'{file_names[i]} empty box')

    for file_name, bbox in tqdm(zip(file_names, bboxes), desc='preds formatting'):
        boxes = np.array(str(bbox).strip().split(' '))

        # boxes - class ID confidence score xmin ymin xmax ymax
        if len(boxes) % 6 == 0:
            boxes = boxes.reshape(-1, 6)
        elif isinstance(bbox, float):
            print(f'{file_name} empty box')
            continue
        else: # else에 걸리는 경우는 모델이 물체 탐지를 못한 경우?
            continue # 학습이 덜 되었다면 물체를 탐지하지 못할 수도 있음 -> 무시
            # raise Exception('error', 'invalid box count')
        for box in boxes:
            new_pred.append([file_name, box[0], box[1], float(box[2]), float(box[4]), float(box[3]), float(box[5])])


    # gt 파일을 불러와서, pascal voc format으로 변환하기 -> CustomDataset의 특성 때문에, 계산이 이상하게 될 수도 있음
    gt = []

    for image_id in imgIDs:
        image_info = coco.loadImgs(image_id)[0]
        annotation_id = coco.getAnnIds(imgIds=image_info['id'])
        annotation_info_list = coco.loadAnns(annotation_id)
            
        file_name = image_info['file_name']
            
        for annotation in annotation_info_list:
            gt.append([file_name, annotation['category_id'],
                    float(annotation['bbox'][0]),
                    float(annotation['bbox'][0]) + float(annotation['bbox'][2]),
                    float(annotation['bbox'][1]),
                    (float(annotation['bbox'][1]) + float(annotation['bbox'][3]))])
    
    # format을 통일한 gt와 preds를 가지고 mAP 계산.
    mean_ap, average_precisions = mean_average_precision_for_boxes(gt, new_pred, iou_threshold=score_threshold)

    return mean_ap, average_precisions


def inference_fn(test_data_loader, model, device):
    outputs = []
    for images in tqdm(test_data_loader):
        # gpu 계산을 위해 image.to(device)
        images = list(image.to(device) for image in images)
        output = model(images)
        for out in output:
            outputs.append({'boxes': out['boxes'].tolist(), 'scores': out['scores'].tolist(), 'labels': out['labels'].tolist()})
    return outputs


# 테스트 코드
if __name__ == "__main__":
    print(load_config("./configs/fasterrcnn.yaml"))
