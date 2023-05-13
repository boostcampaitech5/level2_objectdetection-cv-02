from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
# faster rcnn model이 포함된 library
import torchvision

from torch.utils.data import DataLoader
import datetime
import pandas as pd
from tqdm import tqdm

from my_dataset import TestDataset
from utils import load_config, inference_fn, get_device, get_save_folder_name
from model.baseline_model import get_fasterrcnn_resnet50_fpn



def inference(config_path, model_path, save_folder_path):
    save_file_name = config_path.split('.')[1].split('/')[-1]
    config = load_config(config_path)

    annotation = config['path']['test_annotation'] # annotation 경로
    data_dir = config['path']['image_dir'] # dataset 경로
    test_dataset = TestDataset(annotation, data_dir)
    score_threshold = config['test']['score_threshold']
    check_point = model_path
    

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )
    device = get_device()
    print(device)
    
    # 학습한 모델 불러오기
    model = get_fasterrcnn_resnet50_fpn()
    model.to(device)
    model.load_state_dict(torch.load(check_point))
    model.eval()
    
    outputs = inference_fn(test_data_loader, model, device)
    prediction_strings = []
    file_names = []
    coco = COCO(annotation)

    # submission 파일 생성
    for i, output in enumerate(tqdm(outputs, desc='generate submission.csv')):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > score_threshold: 
                # label[1~10] -> label[0~9]
                prediction_string += str(label-1) + ' ' + str(score) + ' ' + str(box[0]) + ' ' + str(
                    box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    
    print("Save a new submission...", end=' ')
    save_path = f'./submission/{save_folder_path}/{save_file_name}_submission.csv'
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
            
    submission.to_csv(save_path, index=None)
    print(submission.head())
    print("Done!")


if __name__ == "__main__":
    # 1. 모델 config가 담겨있는 yaml 파일 경로 정의
    config_path = "./configs/fasterrcnn4_adam.yaml"

    # 2. 학습한 모델의 weight 파일 경로 지정
    model_weight = '/opt/ml/baseline/level2_objectdetection-cv-02/main/torchvision/checkpoints/2023-5-13/fasterrcnn4_adam_checkpoints.pth' # 체크포인트 경로

    # 3. inference 결과를 저장할 폴더 지정
    save_folder_path = get_save_folder_name()

    # 4. inference
    inference(config_path, model_weight, save_folder_path)
