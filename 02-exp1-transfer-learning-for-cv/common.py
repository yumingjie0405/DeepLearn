# -*- coding: utf-8 -*-
# @Time : 2023/3/16 9:14
# @Author : XuSukui
# @Email : skxu2@iflytek.com
# @File : common.py

# License: BSD
# Author: Sasank Chilamkurthy

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Data augmentation and normalization for training
# Just normalization for validation
# 数据增强
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/hymenoptera_data'
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val']
}
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=8, shuffle=True)
    for x in ['train', 'val']
}
'''
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=4,
                                          shuffle=True, num_workers=4),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=4,
                                          shuffle=False, num_workers=4)
}

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25, dataloaders=dataloaders)
'''
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(class_names)
class_id_name = image_datasets['train'].class_to_idx
# print(class_id_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
