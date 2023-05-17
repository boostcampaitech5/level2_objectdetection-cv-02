import torch
from torch.utils.data import DataLoader

import argparse
import wandb
import datetime
from pprint import pprint

from my_optimizer import MyOptimizer
from transform import get_train_transform, get_valid_transform
from utils import seed_everything, load_config, collate_fn, get_device, get_save_folder_name
from trainer.faster_rcnn_trainer import train_fn
import model.baseline_model as baseline_model
from my_dataset import CustomDataset


def parse_args():
    """터미널 상에서 사용자가 입력한 argument를 저장하는 함수.

    Returns:
        _type_: 사용자가 입력한 argument를 반환
    """
    parser = argparse.ArgumentParser(description='Object Detection Train based on torchvision')

    # parser
    parser.add_argument('--config_path', type=str, default='./configs/fasterrcnn.yaml', help='select config path (default : ./configs/fasterrcnn.yaml)')
    
    args = parser.parse_args()

    return args


def get_dataloader(configs: str):
    """yaml 파일에 정의되어있는 경로를 바탕으로, dataset을 만들고 dataloader를 반환해주는 함수.

    Args:
        configs (str): ./configs/ 아래에 정의해둔 yaml 파일을 load_config() 함수로 불러오고, 이를 넘겨주어야 함.

    Returns:
        torch.utils.data.DataLoader : train과 validation dataloader를 반환함
    """
    print(f'Default setting...', end='')
    # 데이터셋 불러오기
    train_annotation = configs['path']['train_annotation']
    valid_annotation = configs['path']['valid_annotation']

    data_dir = configs['path']['image_dir']

    train_dataset = CustomDataset(train_annotation, data_dir, get_train_transform(configs['augmentation']))
    valid_dataset = CustomDataset(valid_annotation, data_dir, get_valid_transform(configs['augmentation']))

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
    # wandb 세팅
    wandb.init(project="object-detection-torchvision", reinit=True)
    
    # args 설정
    args = parse_args()

    # yaml config 파일 가져오기
    configs = load_config(args.config_path)
    pprint(configs)

    # wandb에 config 업로드하기 (running name 추가, 기록 설정해야 함)
    wandb.config.update(configs)

    # 기본 환경 세팅
    # 1. seed 세팅
    seed_everything(configs['setting']['seed'])

    # 2. device 지정
    device = get_device()   
    
    # 3. 데이터로더 세팅
    train_data_loader, valid_data_loader = get_dataloader(configs)

    # 4. 모델 세팅 (변경 예정)
    model = baseline_model.get_fasterrcnn_resnext101_32x8d_fpn()
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    # 5. optimizer 세팅
    opt_cfg = configs['optimizer']
    my_opt = MyOptimizer(params, opt_cfg)
    my_opt = my_opt()
        
    # 6. 학습한 모델을 저장할 경로 설정
    save_file_name = args.config_path.split('.')[1].split('/')[-1]
    save_folder_name = get_save_folder_name()

    # 7. wandB 실험 이름 설정
    run_name = f"{configs['setting']['who']}-{save_file_name}"
    wandb.run.name = run_name # e.g., sy-fasterrcnn
    
    # 학습 시작!
    train_fn(configs, train_data_loader, valid_data_loader,
             my_opt, model, device, save_folder_name, save_file_name)