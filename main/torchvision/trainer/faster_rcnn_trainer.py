import torch

import os
from tqdm import tqdm

from utils import Averager
from evaluation import evaluate


def train_fn(num_epochs, train_data_loader, val_data_loader,
             optimizer, model, device, save_folder_path):
    best_mean_ap = 0.
    train_loss_hist = Averager()

    for epoch in range(num_epochs):
        
        # train_loop
        model.train()
        train_loss_hist.reset()
        for images, targets, image_ids in tqdm(train_data_loader, desc='training'):

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

        # validation_loop
        mean_ap, ap = evaluate(val_data_loader, model, device)

              
        # validation set의 mAP 기준으로 모델을 저장
        if mean_ap > best_mean_ap:
            print(f"Previous best mAP: {best_mean_ap}, new mAP: {mean_ap} ({mean_ap - best_mean_ap} improve)")
            print("Save a new model...")
            save_path = f'./checkpoints/{save_folder_path}/faster_rcnn_torchvision_checkpoints.pth'
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            torch.save(model.state_dict(), save_path)
            print("Done!")
            best_mean_ap = mean_ap
