# -*- coding: utf-8 -*-
# @Time : 2023/3/16 9:36
# @Author : XuSukui
# @Email : skxu2@iflytek.com
# @File : finetune.py
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler

from common import device
from utils import train_model, visualize_model
#TODO
model_ft = models.resnet18(pretrained=True)

num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

torch.save(model_ft.state_dict(), 'model_finetune.pth')
visualize_model(model_ft)
input()

# You can try to save and reload your best model on local disk
# You can try to test the model and analysis the results by your own code
# You can use your own special visualize code
