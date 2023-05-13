import torch
import wandb

import os
from tqdm import tqdm

from utils import Averager
from evaluation import evaluate


def train_fn(cfgs: dict, train_data_loader, val_data_loader,
             optimizer, model, device, save_folder_name: str, save_file_name: str):
    """torchvision의 object detection 모델을 학습시킬 때 사용하는 함수.

    Args:
        cfgs (dict): 사전에 정의한 yaml 파일을 불러온 dict 형식의 설정 값들.
        train_data_loader (torch.utils.data.DataLoader): train dataloader.
        val_data_loader (torch.utils.data.DataLoader): validation dataloader.
        optimizer (_type_): 가중치 업데이트에 사용할 optimizer.
        model (_type_): 학습시킬 object detection 모델.
        device (_type_): 학습에 사용할 장비 e.g., cuda, cpu
        save_folder_name (str): 학습 결과를 저장할 폴더의 이름. 이후 저장 경로를 지정할 때 사용됨.
        save_file_name (str): 학습 결과를 저장할 파일의 이름. yaml 파일의 이름을 바탕으로 설정됨.
    """

    best_mean_ap = 0.
    train_loss_hist = Averager()

    
    for epoch in range(cfgs['hparams']['epochs']):
        
        # train_loop
        model.train()
        train_loss_hist.reset()
        description = f"Epoch #{epoch+1} training"
        
        for images, targets, image_ids in tqdm(train_data_loader, desc=description):

            # gpu 계산을 위해 image.to(device)
            images = list(image.float().to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # calculate loss
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            train_loss_hist.send(loss_value)

            # backward
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        
        print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value}")

        # wandb logging - train
        wandb.log({
            "Epoch": epoch+1,
            "Train/loss": train_loss_hist.value,
        })
        
        # validation_loop
        mean_ap, ap = evaluate(val_data_loader, model, cfgs, device)

        # wandb logging - validation
        wandb.log({
            "Epoch": epoch+1,
            "mAP50": mean_ap,
            "AP-General trash": ap['0'][0],
            "AP-Paper": ap['1'][0],
            "AP-Paper pack": ap['2'][0],
            "AP-Metal": ap['3'][0],
            "AP-Glass": ap['4'][0],
            "AP-Plastic": ap['5'][0],
            "AP-Styrofoam": ap['6'][0],
            "AP-Plastic bag": ap['7'][0],
            "AP-Battery": ap['8'][0],
            "AP-Clothing": ap['9'][0],
        })

              
        # validation set의 mAP 기준으로 모델을 저장
        if mean_ap > best_mean_ap:
            print(f"Previous best mAP: {best_mean_ap}, new mAP: {mean_ap} ({mean_ap - best_mean_ap} improve)")
            print("Save a new model...")
            save_path = f'./checkpoints/{save_folder_name}/{save_file_name}_checkpoints.pth'
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            torch.save(model.state_dict(), save_path)
            print("Done!")
            best_mean_ap = mean_ap
