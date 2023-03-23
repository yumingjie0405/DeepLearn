# -*- coding: utf-8 -*-
# @Time : 2023/3/15 13:48
# @Author : XuSukui
# @Email : skxu2@iflytek.com
# @File : plt_train.py

import os

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl

from dataset import CustomImageDataset

learning_rate = 1e-3
batch_size = 64
epochs = 100

class NeuralNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        X, y = batch
        pred = self.forward(X)
        loss = F.cross_entropy(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        pred = self.forward(X)
        loss = F.cross_entropy(pred, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        return optimizer

model = NeuralNetwork()
trainer = pl.Trainer(max_epochs=epochs)

base_data_dir = os.path.join('data', 'FashionMNIST', 'images')
train_annotations_file = os.path.join(base_data_dir, 'train_label.csv')
train_img_dir = os.path.join(base_data_dir, 'train')
test_annotations_file = os.path.join(base_data_dir, 'test_label.csv')
test_img_dir = os.path.join(base_data_dir, 'test')

training_data = CustomImageDataset(annotations_file=train_annotations_file, img_dir=train_img_dir)
test_data = CustomImageDataset(annotations_file=test_annotations_file, img_dir=test_img_dir)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
