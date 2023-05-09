import os
import copy
import torch
from tqdm import tqdm
import pandas as pd
import argparse
import numpy as np
import random

import detectron2
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from mapper import InferenceMapper

#fixed seed
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def inference(save_dir, model_name, args):
    seed_everything(args.seed) #fixed seed

    # Register Dataset
    try:
        register_coco_instances('coco_trash_test', {}, '../../dataset/test.json', args.data_dir)
    except AssertionError:
        pass

    # config 불러오기
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f'COCO-Detection/{model_name}.yaml'))

    # config 수정하기
    cfg.DATASETS.TEST = ('coco_trash_test',)

    cfg.DATALOADER.NUM_WOREKRS = 2

    cfg.OUTPUT_DIR = save_dir

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.model_file_name)

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3

    # model
    predictor = DefaultPredictor(cfg)


    # test loader
    test_loader = build_detection_test_loader(cfg, 'coco_trash_test', InferenceMapper)

    prediction_strings = []
    file_names = []

    class_num = 10

    for data in tqdm(test_loader):
        
        prediction_string = ''
        
        data = data[0]
        
        outputs = predictor(data['image'])['instances']
        
        targets = outputs.pred_classes.cpu().tolist()
        boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
        scores = outputs.scores.cpu().tolist()
        
        for target, box, score in zip(targets,boxes,scores):
            prediction_string += (str(target) + ' ' + str(score) + ' ' + str(box[0]) + ' ' 
            + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ')
        
        prediction_strings.append(prediction_string)
        file_names.append(data['file_name'].replace(args.data_dir,''))

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(cfg.OUTPUT_DIR, f'{model_name}.csv'), index=None)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Obejct Detection inference by Detectron2')

    #parser 
    parser.add_argument('--seed', type=int, default=42, help='Fixed Seed (default : 42)')
    parser.add_argument('--data_dir', type=str, default='../../dataset/', help='inference data dir (default : ../../dataset/)')
    
    parser.add_argument('--save_dir', type=str, default='./save/')
    parser.add_argument('--model', type=str, default='faster_rcnn_R_101_FPN_3x', help='train model name (default : faster_rcnn_R_101_FPN_3x)')
    parser.add_argument('--model_file_name', type=str, default='model_final.pth', help='load model .pth file name')

    args = parser.parse_args()

    save_dir = os.path.join(args.save_dir, args.model)
    model_name = args.model
    inference(save_dir, model_name, args)



