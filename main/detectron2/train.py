import os
import copy
import torch
import argparse
import numpy as np
import random


import detectron2

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

from trainer import MyTrainer



#fixed seed
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

#train
def train(save_dir, model_name, args):
    seed_everything(args.seed) #fixed seed

    #Register Dataset
    try:
        register_coco_instances('coco_trash_train', {}, args.train_json, args.data_dir)
    except AssertionError:
        pass

    try:
        register_coco_instances('coco_trash_val', {}, args.val_json, args.data_dir)
    except AssertionError:
        pass

    MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
                                                         "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

    #config setting
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f'COCO-Detection/{model_name}.yaml'))

    cfg.DATASETS.TRAIN = ('coco_trash_train',)
    cfg.DATASETS.TEST = ('coco_trash_val',)

    cfg.DATALOADER.NUM_WOREKRS = 2

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 15000
    cfg.SOLVER.STEPS = (8000,12000)
    cfg.SOLVER.GAMMA = 0.005
    cfg.SOLVER.CHECKPOINT_PERIOD = 3000

    cfg.OUTPUT_DIR = save_dir

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

    cfg.TEST.EVAL_PERIOD = 500

    
    #save config
    # with open(f"{save_dir}/{model_name}.yaml", "w") as f:
    #     f.write(cfg.dump())
    #     print('Save Config')

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Obejct Detection Train by Detectron2')

    #parser 
    parser.add_argument('--seed', type=int, default=42, help='Fixed Seed (default : 42)')
    parser.add_argument('--data_dir', type=str, default='../../dataset/', help='Train data dir (default : ../../dataset/)')
    parser.add_argument('--train_json', type=str, default='./json/detectron2_train.json', help='train.json dir (default : ./json/detectron2_train.json)')
    parser.add_argument('--val_json', type=str, default='./json/detectron2_val.json', help='val.json dir (default : ./json/detectron2_val.json)')

    parser.add_argument('--save_dir', type=str, default='./save/')
    parser.add_argument('--model', type=str, default='faster_rcnn_R_101_FPN_3x', help='train model name (default : faster_rcnn_R_101_FPN_3x)')

    args = parser.parse_args()

    save_dir = os.path.join(args.save_dir, args.model)
    model_name = args.model

    #저장경로 없다면 생성
    os.makedirs(save_dir, exist_ok=True)

    print('-'*80)
    print('| Save_dir | :' , save_dir)
    print('| Use model | : ', model_name)
    print('-'*80)
    print('train start!!')
    print()


    train(save_dir, model_name, args)



