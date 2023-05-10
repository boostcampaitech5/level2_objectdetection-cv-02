import numpy as np
import random
import torch
import os 

from detectron2 import model_zoo


def seed_everything(seed):
    """_summary_

    Args:
        seed (int): seed 번호 
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


#model config 설정
def train_config_setting(cfg:dict, args:dict, save_dir:str):
    """_summary_

    Args:
        cfg (dict): config
        args (dict): args
        save_dir (str): config 파일 저장경로

    Returns:
        cfg : 변경된 config 파일 리턴
    """
    cfg.merge_from_file(model_zoo.get_config_file(f'{args.config_path}/{args.model}.yaml'))

    cfg.DATASETS.TRAIN = ('coco_trash_train',)
    cfg.DATASETS.TEST = ('coco_trash_val',)

    cfg.DATALOADER.NUM_WOREKRS = 2

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f'{args.config_path}/{args.model}.yaml')
    cfg.MODEL.MASK_ON = False
    cfg.SOLVER.IMS_PER_BATCH = 5

    #epochs를 maxiter로 변환
    epochs = args.epochs
    max_iter = int(4474 / cfg.SOLVER.IMS_PER_BATCH * epochs) #4474 : Image Data num

    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = (2000, 4000)
    cfg.SOLVER.GAMMA = 0.005
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000

    cfg.OUTPUT_DIR = save_dir

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

    cfg.TEST.EVAL_PERIOD = 1000

    #save config
    with open(f"{save_dir}/{args.model}.yaml", "w") as f:
        f.write(cfg.dump())

    return cfg

def inference_config_setting(cfg:dict, args:dict, save_dir:str):
    """_summary_

    Args:
        cfg (dict): config
        args (dict): args
        save_dir (str): config 파일 저장경로

    Returns:
        cfg : 변경된 config 파일 리턴
    """
    cfg.merge_from_file(model_zoo.get_config_file(f'{args.config_path}/{args.model}.yaml'))

    cfg.DATASETS.TEST = ('coco_trash_test',)

    cfg.DATALOADER.NUM_WOREKRS = 2

    cfg.OUTPUT_DIR = save_dir

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.model_file_name)

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3

    return cfg