import os
import copy
import torch
import argparse
import numpy as np
import random
import datetime
import wandb

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

from trainer import MyTrainer
from utils import seed_everything
from utils import train_config_setting


def parse_args():
    """_summary_

    Returns:
        args : args 설정값 
    """
    parser = argparse.ArgumentParser(description='Obejct Detection Train by Detectron2')

    #parser 
    parser.add_argument('--seed', type=int, default=42, help='Fixed Seed (default : 42)')
    parser.add_argument('--data_dir', type=str, default='../../dataset/', help='Train data dir (default : ../../dataset/)')
    parser.add_argument('--train_json', type=str, default='./json/detectron2_train.json', help='train.json dir (default : ./json/detectron2_train.json)')
    parser.add_argument('--val_json', type=str, default='./json/detectron2_val.json', help='val.json dir (default : ./json/detectron2_val.json)')

    parser.add_argument('--save_dir', type=str, default='./save/')
    parser.add_argument('--name', type=str, default='ho', help="spelling your name (default : ho)"  )
    parser.add_argument('--config_path', type=str, default='COCO-Detection', help='select config path (default : COCO-Detection/)')
    parser.add_argument('--model', type=str, default='faster_rcnn_R_101_FPN_3x', help='train model name (default : faster_rcnn_R_101_FPN_3x)')
    parser.add_argument('--epochs', type=int, default=10, help='train epochs (default = 10)')
    args = parser.parse_args()
    return args

#train
def train(save_dir:str, args:dict):
    """_summary_

    Args:
        save_dir (str): 학습 저장경로(모델, config 등)
        args (dict): argsparser 
    """
    wandb.login()
    
    wandb.init(project='detectron2', sync_tensorboard=True)
    wandb.run.name =f"{args.name}-{args.model}"
    
    #fixed seed
    seed_everything(args.seed) 

    #Register Dataset : train과 val dataset 등록
    for mode in ['train', 'val']:
        if mode == 'train':
            json_path = args.train_json
        else:
            json_path = args.val_json

        try:
            register_coco_instances(f'coco_trash_{mode}', {}, json_path, args.data_dir)
        except AssertionError:
            pass

    MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
                                                        "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

    #default config 생성
    cfg = get_cfg()

    #config 값 변경
    cfg = train_config_setting(cfg, args, save_dir)


    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    #args 설정
    args = parse_args()

    #학습 날짜
    now = datetime.datetime.now()
    save_time = now.strftime('%Y-%m-%d')
    
    #저장경로 설정 및 없다면 경로 생성
    save_dir = os.path.join(args.save_dir, args.model, save_time)
    os.makedirs(save_dir, exist_ok=True)

    print('-'*80)
    print('| Save_dir | :' , save_dir)
    print('| Use model | : ', args.model)
    print('-'*80)
    print('train start!!')
    print()

    #train
    train(save_dir, args)



