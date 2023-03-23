# -*- coding: utf-8 -*-
# @Time : 2023/3/15 9:08
# @Author : XuSukui
# @Email : skxu2@iflytek.com
# @File : dataset.py

import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        image = (image / 255).type(torch.float32)
        label = self.img_labels.iloc[idx, 1]
        return image, label

if __name__ == '__main__':
    base_data_dir = os.path.join('data', 'FashionMNIST', 'images')
    train_annotations_file = os.path.join(base_data_dir, 'train_label.csv')
    train_img_dir = os.path.join(base_data_dir, 'train')
    test_annotations_file = os.path.join(base_data_dir, 'test_label.csv')
    test_img_dir = os.path.join(base_data_dir, 'test')

    training_data = CustomImageDataset(annotations_file=train_annotations_file, img_dir=train_img_dir)
    test_data = CustomImageDataset(annotations_file=test_annotations_file, img_dir=test_img_dir)

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")
