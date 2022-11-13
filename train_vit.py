from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision
from tqdm import tqdm
from vit_pytorch import ViT

BASE_CKPT_PATH = "/home/vp.shivasan/IvT/SparseAttentionViT/checkpoints/"

BATCH_SIZE_TRAIN = 16
BATCH_SIZE_TEST = 64
IMAGE_SIZE = 256
transform_train = torchvision.transforms.Compose([torchvision.transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                                                    transforms.RandomResizedCrop(IMAGE_SIZE),
                                                    transforms.RandomHorizontalFlip(),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

transform_test = torchvision.transforms.Compose([torchvision.transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                                                    transforms.CenterCrop(IMAGE_SIZE),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
train_set = torchvision.datasets.ImageFolder(root="/home/vp.shivasan/IvT/data/imagenette2/train",transform=transform_train)
test_set = torchvision.datasets.ImageFolder(root="/home/vp.shivasan/IvT/data/imagenette2/val",transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True,num_workers=2)

valid_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True,num_workers=2)

LOG_NAME = "test"
MODEL_NAME = "original_vit"
model = ViT(
    image_size = 256,
    patch_size = 16,
    num_classes = 10,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

epochs = 50
lr = 3e-5
gamma = 0.7
seed = 42
device = "cuda:1"
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)


for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )

torch.save(model.state_dict(), BASE_CKPT_PATH + MODEL_NAME +"/" + LOG_NAME + ".pt")




