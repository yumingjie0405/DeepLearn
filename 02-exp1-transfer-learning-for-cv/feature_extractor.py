# -*- coding: utf-8 -*-
# @Time : 2023/3/16 10:00
# @Author : XuSukui
# @Email : skxu2@iflytek.com
# @File : feature_extractor.py

import torch
import torch.nn as nn
import torchvision

import torch.optim as optim
from torch.optim import lr_scheduler

from common import device
from utils import train_model, visualize_model

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
# optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
# TODO3 使用Adm随机优化器
optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=0.001)
# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_conv, T_max=10)
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)

torch.save(model_conv.state_dict(), 'model_feature_extractor.pth')
visualize_model(model_conv)
input()
