
import os
import shutil

import numpy as np
import pandas as pd

import torch
from torchvision.io import write_jpeg
from torchvision import datasets
from torchvision.transforms import ToTensor

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

def write_mnist_dataset(training=True, root='data'):
    data = datasets.FashionMNIST(
        root=root,
        train=training,
        download=True,
        transform=ToTensor()
    )

    img_dir = 'train' if training else 'test'
    img_dir = os.path.join(root, 'FashionMNIST', 'images', img_dir)
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    os.makedirs(img_dir)

    tags = np.zeros(10, dtype=np.int32)
    img_files = list()
    labels = list()
    if training:
        label_csv = os.path.join(root, 'FashionMNIST', 'images', 'train_label.csv')
    else:
        label_csv = os.path.join(root, 'FashionMNIST', 'images', 'test_label.csv')
    for k in range(len(data)):
        img, label = data[k]
        tags[label] += 1
        image_filename = labels_map[label] + str(tags[label]) + '.jpg'
        img_files.append(image_filename)
        labels.append(label)
        write_jpeg((img * 255).type(torch.uint8), os.path.join(img_dir, image_filename))
    df = pd.DataFrame({'img_filename': img_files, 'labels': labels})
    df.to_csv(label_csv, index=False, header=False, decimal=',')

if __name__ == '__main__':
    print("生成训练集")
    write_mnist_dataset()
    print("生成测试集")
    write_mnist_dataset(training=False)
