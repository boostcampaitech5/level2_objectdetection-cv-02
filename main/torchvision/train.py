import torch
from torch.utils.data import DataLoader

import argparse
import wandb
import datetime

import my_optimizer
from transform import get_train_transform, get_valid_transform
from utils import seed_everything, load_config, collate_fn, get_device
from trainer.faster_rcnn_trainer import train_fn
from model.baseline_model import get_fasterrcnn_resnet50_fpn
from my_dataset import CustomDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection Train by torchvision')

    # parser
    parser.add_argument('--config_path', type=str, default='./configs/fasterrcnn.yaml', help='select config path (default : ./configs/fasterrcnn.yaml)')
    
    args = parser.parse_args()

    return args


def get_data(configs: str):
    print(f'Default setting...', end='')
    # 데이터셋 불러오기
    train_annotation = configs['path']['train_annotation']
    valid_annotation = configs['path']['valid_annotation']

    data_dir = configs['path']['image_dir']

    train_dataset = CustomDataset(train_annotation, data_dir, get_train_transform())
    valid_dataset = CustomDataset(valid_annotation, data_dir, get_valid_transform())

    # 데이터로더 생성
    train_data_loader= DataLoader(
        train_dataset,
        batch_size=configs['train']['batch_size'],
        shuffle=configs['train']['shuffle'],
        num_workers=configs['train']['num_workers'],
        collate_fn=collate_fn
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=configs['valid']['batch_size'],
        shuffle=configs['valid']['shuffle'],
        num_workers=configs['valid']['num_workers'],
    )


    return train_data_loader, valid_data_loader


if __name__ == "__main__":
    wandb.init(project="object-detection-torchvision", reinit=True)
    
    today = datetime.datetime.now()
    save_folder_path = f'{today.year}-{today.month}-{today.day}'

    # args 설정
    args = parse_args()

    # yaml config 파일 가져오기
    configs = load_config(args.config_path)

    # seed 세팅
    seed_everything(configs.seed)

    # device 지정
    device = get_device()

    
    
    # 기본 환경 세팅
    # 1. 데이터로더 세팅
    train_data_loader, valid_data_loader = get_data(configs)

    # 2. 모델 세팅
    model = get_fasterrcnn_resnet50_fpn()
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    # 3. optimizer 세팅
    opt_cfg = configs['hparams']['optimizer']

    my_opt = None
    if opt_cfg['name'] == 'sgd':
        my_opt = my_optimizer.get_sgd(params,
                                      opt_cfg['lr'],
                                      opt_cfg['momentum'],
                                      opt_cfg['weight_decay'])
    
    # 학습 시작!
    train_fn(configs['hparams']['epochs'],
             train_data_loader, valid_data_loader,
             my_opt, model, device, save_folder_path)