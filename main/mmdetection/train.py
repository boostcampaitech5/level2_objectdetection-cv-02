# 모듈 import

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device
import argparse
import os
import wandb

wandb.init(project="mmdetection") #나중에 팀 프로젝트로 바꿔주기

def wandb_config(cfg,args):
    config_dict  = {'seed'         : args.seed,
                    'config'       : args.config,
                    'output_dir'   : args.output_dir,
                    'model_type'   : cfg.model.type,
                    'backbone'     : cfg.model.backbone.type,
                    'neck'         : cfg.model.neck.type,
                    'image_scale'  : cfg.data.train.pipeline[2]['img_scale']      
                    }
    return config_dict

def modify_config(cfg, args):
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    root = '../../dataset/'
    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
            init_kwargs={'project': 'MMDetection-tutorial'},
            interval=10,
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=0)]

    output_root = '/opt/ml/baseline/mmdetection/work_dirs/'
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = root + 'train.json' # train json 정보
    cfg.data.train.pipeline[2]['img_scale'] = (1024,1024) # Resize

    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = root
    cfg.data.val.ann_file = root + 'val.json' # validation json 정보
    cfg.data.val.pipeline[1]['img_scale'] = (1024,1024) # Resize

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json' # test json 정보
    cfg.data.test.pipeline[1]['img_scale'] = (1024,1024) # Resize

    cfg.data.samples_per_gpu = 4
    output_dir = args.output_dir
    if not os.path.exists(output_root + output_dir):
        os.makedirs(output_root + output_dir) # 저장 경로 없으면 생성
    cfg.work_dir = output_root + output_dir # 저장 경로 변경
   
    cfg.seed = args.seed
    cfg.gpu_ids = [0]
    no_roi_head = ['RetinaNet', 'TOOD']
    if cfg.model.type in no_roi_head : #roi head가 없으면
        cfg.model.bbox_head.num_classes = 10 
    
    else : #일단 그 외는 아래로 정의. cfg 하나씩 보고 조건 추가해주자
        if isinstance(cfg.model.roi_head.bbox_head, list):
            for i in range(len(cfg.model.roi_head.bbox_head)):
                cfg.model.roi_head.bbox_head[i].num_classes = 10 
        else : #그냥 딕셔너리면
            cfg.model.roi_head.bbox_head.num_classes = 10 
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()
    return cfg

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', required=True, help='Please write down .py as the absolute path')
    parser.add_argument('--seed', type=int, default=2022, help='default : 2022')
    parser.add_argument('--output_dir', required=True, type=str, help='Please specify the desired directory name')
    parser.add_argument('--exp_name', type=str, default='dahyeon', help='wandb exp name (default: dahyeon)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)  
    modify_config(cfg, args)  
    model = build_detector(cfg.model)
    model.init_weights() 
    datasets = [build_dataset(cfg.data.train)]
    wandb.config = wandb_config(cfg, args)
    train_detector(model, datasets[0], cfg, distributed=False, validate=False)

## 경우 1 : faster_cnn
## python train.py --config /opt/ml/baseline/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py --output_dir test_faster
## 경우 2 : cascade_rcnn
## python train.py --config /opt/ml/baseline/mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py --output_dir test_cas
## 경우 3 : retinanet
## python train.py --config /opt/ml/baseline/mmdetection/configs/retinanet/retinanet_r50_caffe_fpn_mstrain_3x_coco.py --output_dir test_retina 
## 경우 4 : TOOD
## python train.py --config /opt/ml/baseline/mmdetection/configs/tood/tood_r50_fpn_anchor_based_1x_coco.py --output_dir test_tood