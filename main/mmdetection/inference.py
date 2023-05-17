import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np
import argparse

def modify_config(cfg, args):
    """
    Modify the configuration file based on the arguments passed in.

    Args:
        cfg (Config): A configuration object that contains information
        args: The command line arguments

    Returns:
        The modified configuration file
    """
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    root='/opt/ml/dataset/'
    cfg = Config.fromfile(args.cfg_file)
    pth_name = args.pth_name 
    cfg.work_dir = args.work_dir

    no_roi_head = ['RetinaNet','PPA','YOLOV3', 'DeformableDETR' ,'TOOD', 'GFL']
    if cfg.model.type in no_roi_head : 
        cfg.model.bbox_head.num_classes = 10 
    else : 
            if isinstance(cfg.model.roi_head.bbox_head, list):
                for i in range(len(cfg.model.roi_head.bbox_head)):
                    cfg.model.roi_head.bbox_head[i].num_classes = 10 
            else : 
                cfg.model.roi_head.bbox_head.num_classes = 10   

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json'
    cfg.data.test.pipeline[1]['img_scale'] = (1024,1024) # Resize
    cfg.data.test.test_mode = True

    cfg.data.samples_per_gpu = 4 #batch_size

    cfg.seed=2022
    cfg.gpu_ids = [1]
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.model.train_cfg = None
    cfg.model.test_cfg.score_thr = args.confidence_thr
    
    return cfg

def data_load(cfg):
    """
    Build a test dataset and data loader based on the configuration.

    Args:
        cfg (Config): A configuration object that contains information

    Returns:
        tuple: A tuple containing the test dataset
    """
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )
    checkpoint_path = os.path.join(cfg.work_dir, f'{args.pth_name}.pth')
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    if args.nms_iou_thr:
        model.test_cfg.nms.iou_threshold = args.nms_iou_thr
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])
    output = single_gpu_test(model, data_loader, show_score_thr=args.show_score_thr) 
    return output

def create_submission(output,cfg,args):
    """
    Create a submission file in CSV format, with predicted bounding boxes and scores for each test image. 
    The submission file is saved to the work directory.
    Args:
        output (list): A predicted bounding boxes and scores for each test image
        cfg (Config): A configuration object that contains information
        args (argparse.Namespace): The command line arguments

    Returns:
        submission.csv
    """
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)
    
    img_ids = coco.getImgIds()
    class_num = 10
    for i, out in enumerate(output):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(class_num):
            for o in out[j]:
                prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                    o[2]) + ' ' + str(o[3]) + ' '
            
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(cfg.work_dir, f'{args.submission}.csv'), index=None)
    

def parse_args():
    """
    Parses command line arguments for training a detector.

    Returns:
        args (argparse.Namespace): The command line arguments
    """
    parser = argparse.ArgumentParser(description='save submission files')
    parser.add_argument('--cfg_file', required=True, help='Path to the configuration file for the detector')
    parser.add_argument('--work_dir', required=True, help='Directory to save the output files (default: 2022)')
    parser.add_argument('--pth_name', required=True, help='Name of the PyTorch checkpoint file to save the trained model')
    parser.add_argument('--submission', required=True, help='Directory name to save the submission files and to log results with Weights & Biases')
    parser.add_argument('--confidence_thr', type=float, default=0.05, help='confidence threshold value when you make a submission.csv (default: 0.05)') # confidence score threshold
    parser.add_argument('--nms_iou_thr', type=float, help='IoU threshold used in NMS') #검출된 객체의 중복을 제거

    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    """Run object detection on a given dataset and generate a submission file.

    Args:
        cfg_file (str): Path to the config file.
        works_dir (str): Path to the worked directory.
        pth_name (str): Name of the trained model checkpoint.
        submission (str): Name of the submission file to generate.

    Returns:
        submission file
    """
    args = parse_args()
    cfg = Config.fromfile(args.cfg_file)  
    cfg = modify_config(cfg, args) 
    output = data_load(cfg) 
    create_submission(output,cfg,args)
    print("-----------------------------------------------------")
    print("Successful! Go to {}.".format(cfg.work_dir))
    print("-----------------------------------------------------")

# Please place the inference.py file at /opt/ml/baseline/mmdetection/<<inference.py>>
# (ex) python inference.py 
#      --cfg_file ./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py
#      --work_dir ./work_dirs/faster_rcnn_x101_32x4d_fpn_1x_coco_trash 
#      --pth_name latest
#      --submission exexexex  
#      --show_score_thr 0.05
#      --nms_iou_thr 0.2
#      
# => result : exexexex.csv
