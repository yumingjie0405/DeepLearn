# -*- coding: utf-8 -*-
# @Time : 2023/3/15 10:20
# @Author : XuSukui
# @Email : skxu2@iflytek.com
# @File : train.py

import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import CustomImageDataset
from model import NeuralNetwork

learning_rate = 1e-3
batch_size = 64
epochs = 100

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # noinspection PyUnresolvedReferences
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


base_data_dir = os.path.join('data', 'FashionMNIST', 'images')
train_annotations_file = os.path.join(base_data_dir, 'train_label.csv')
train_img_dir = os.path.join(base_data_dir, 'train')
test_annotations_file = os.path.join(base_data_dir, 'test_label.csv')
test_img_dir = os.path.join(base_data_dir, 'test')

training_data = CustomImageDataset(annotations_file=train_annotations_file, img_dir=train_img_dir)
test_data = CustomImageDataset(annotations_file=test_annotations_file, img_dir=test_img_dir)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

model = NeuralNetwork().to(device)
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

