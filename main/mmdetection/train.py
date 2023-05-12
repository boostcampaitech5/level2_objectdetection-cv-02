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
import json
from pipeline import get_testpipeline, get_trainpipeline, get_valpipeline

def wandb_config(cfg,args):
    """
      Parameters to save in wandb  
    """
    config_dict  = {
                    'seed'         : args.seed,
                    'config'       : args.config,
                    'exp_name'     : args.exp_name,
                    'model_type'   : cfg.model.type,
                    'backbone'     : cfg.model.backbone.type,
                    'neck'         : cfg.model.neck.type,
                    'epoch'        : cfg.runner.max_epochs,
                    'image_scale'  : cfg.data.test.pipeline[1]['img_scale'], 
                    'batch_size'   : args.batch_size,     
                    }
    return config_dict
def modify_pipeline(cfg, args) :
    """Modify the augmentation pipeline"""

    #Resize img_size
    # auto_resize = ['YOLOV3', 'PanopticFPN','DeformableDETR', 'SparseRCNN'] #이 모델들은 직접 config에서 수정해주어야 함.
    # if cfg.model.type not in auto_resize :
    #     cfg.data.train.pipeline[2]['img_scale'] = (1024,1024)
    #     cfg.data.val.pipeline[1]['img_scale'] = (1024,1024)
    #     cfg.data.test.pipeline[1]['img_scale'] = (1024,1024)

    #train pipeline 
    cfg.data.train.pipeline = get_trainpipeline(args.pipeline, trash_norm = args.trash_norm)

    #validation pipeline
    cfg.data.val.pipeline = get_valpipeline(args.pipeline, trash_norm = args.trash_norm)

    #test pipeline 
    cfg.data.test.pipeline = get_testpipeline(args.pipeline, trash_norm = args.trash_norm)

def modify_config(cfg, args):
    f"""Modify the variable within the config to suit the task
        Parameters : 
            --num_classes : 10
            --classes : ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
            --root : The dir where the train.json and val.json are located
            --output_root : The directory where the output will be saved. 
            --hook : 
            --img_scale : resize img_scale 
            --sample_per_gpu : batch_size
            --output_dir : exp_name
            --no_roi_head : Models without the roi_head in their architecture.
            --auto_resize : not a single resize 
        
        Models : 
            faster_rcnn, cascade_rcnn, retinanet, tood, swin, paa, detr, sparse_rcnn 

    """
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    root = "/opt/ml/dataset/"
    output_root = '/opt/ml/baseline/mmdetection/work_dirs/'

    #hooks 지정
    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
            init_kwargs={'project': 'mmdetection'},
            interval=10,
            log_checkpoint=False,
            log_checkpoint_metadata=False,
            num_eval_images=0)]


    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = root + 'train.json' # train json 정보

    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = root
    cfg.data.val.ann_file = root + 'val.json' # validation json 정보

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json' # test json 정보

    cfg.evaluation.classwise = True
    
    #batch_size 수정
    cfg.data.samples_per_gpu = args.batch_size

    #max_epochs
    if args.epoch :
        cfg.runner.max_epochs = args.epoch

    #저장 경로 설정
    output_dir = args.exp_name
    if not os.path.exists(output_root + output_dir):
        os.makedirs(output_root + output_dir) # 저장 경로 없으면 생성
    cfg.work_dir = output_root + output_dir # 저장 경로 변경
   
    cfg.seed = args.seed
    cfg.gpu_ids = [0]
    
    #model 구조 별로 다르게 config 수정해야 함.
    #roi_head 유무
    roi_head_type = 0
    no_roi_head = ['RetinaNet','PPA','YOLOV3', 'DeformableDETR' ,'TOOD']
    if cfg.model.type in no_roi_head : #roi head가 없으면
        cfg.model.bbox_head.num_classes = 10 
    
    else : #일단 그 외는 아래로 정의. cfg 하나씩 보고 조건 추가해주자
        #roi_head.bbox 
        if isinstance(cfg.model.roi_head.bbox_head, list):
            for i in range(len(cfg.model.roi_head.bbox_head)): 
                cfg.model.roi_head.bbox_head[i].num_classes = 10 
                roi_head_type = 1
        else : #그냥 딕셔너리면 (single bbox)
            cfg.model.roi_head.bbox_head.num_classes = 10 
            roi_head_type = 2

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()
    return cfg

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', required=True, help='Please write down .py as the absolute path')
    parser.add_argument('--seed', type=int, default=2022, help='default : 2022')
    parser.add_argument('--epoch', type=int, help = "max epoches")
    parser.add_argument('--exp_name', default ='test', type=str, help='save dir name and wandb') #저장할 폴더명/wandb 실험명 생성
    parser.add_argument('--batch_size', type=int, default=4, help="batch_size , default = 4")
    parser.add_argument('--myname', type=str, default='who', help = "Your initial, default = who")
    parser.add_argument('--pipeline', type=str, default="BaseAugmentation", help = "Custom Augmentation, default : BaseAugmentation")
    parser.add_argument('--trash_norm', action='store_true', help = "use trash_Datasets mean&std, default : coco")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)  
    modify_config(cfg, args)  
    model = build_detector(cfg.model)
    model.init_weights() 
    datasets = [build_dataset(cfg.data.train)]
    wandb_cfg = wandb_config(cfg,args)
    wandb.init(project="mmdetection", 
               config=wandb_cfg, 
               name= f'{args.myname}-{cfg.model.type}-{args.exp_name}'
               )
    wandb.save("config.json")
    train_detector(model, datasets[0], cfg, distributed=False, validate=True)


## 경우 1 : faster_cnn
## python train.py --config /opt/ml/baseline/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py --exp_name test_faster
## 경우 2 : cascade_rcnn
## python train.py --config /opt/ml/baseline/mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py --exp_name test_cas
## 경우 3 : retinanet
## python train.py --config /opt/ml/baseline/mmdetection/configs/retinanet/retinanet_r50_caffe_fpn_mstrain_3x_coco.py --exp_name test_retina 
## 경우 4 : TOOD
## python train.py --config /opt/ml/baseline/mmdetection/configs/tood/tood_r50_fpn_anchor_based_1x_coco.py --exp_name test_tood
## 경우 5 : swin #뭔가 이상함 일단 보류
## python train.py --config /opt/ml/baseline/mmdetection/configs/swin/retinanet_swin-t-p4-w7_fpn_1x_coco.py --exp_name test_swin


##경우 n : ppa  #exp_name 지정할것
## python train.py --config /opt/ml/baseline/mmdetection/configs/paa/paa_r101_fpn_2x_coco.py --exp_name "ppa"
##경우 n : DeformableDETR
## python train.py --auto-scale-lr --config /opt/ml/baseline/mmdetection/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py --exp_name DETR --epoch 50
##경우 n : SparseRCNN
## python train.py --auto-scale-lr --config /opt/ml/baseline/mmdetection/configs/sparse_rcnn/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py --exp_name sparse_rcnn --epoch 30


## myname : 이름스펠링_modeltype_neck~ 뒤는 알아서 맘대로 
## (ex) dh_cascade_rcnn_fpn_1,2,3...
