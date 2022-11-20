from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile
import torch.distributed as dist
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
from models.big_bird_vit import BigBirdViT
torch.autograd.set_detect_anomaly(True)
import argparse
torch.cuda.set_device(1)

parser = argparse.ArgumentParser('ViT Args', add_help=False)

parser.add_argument('--train_batch_size', default=16, type=int)
parser.add_argument('--test_batch_size', default=64, type=int)
parser.add_argument('--image_size', default=368, type=int)
parser.add_argument('--patch_size', default=16, type=int)
parser.add_argument('--vit_arch', default="OriginalViT", type=str,help="OriginalViT, BigBirdViT, GraphViT")



#Training specs
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--lr', default=3e-5, type=float)
parser.add_argument('--gamma', default=0.7, type=float)

## GPU ARGS ##
parser.add_argument('--use_gpu', default=True, type=bool)
parser.add_argument('--distributed', default=True, type=bool)
parser.add_argument('--local_rank', default=1, type=int)
parser.add_argument('--num_gpus', default=4, type=int)
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--resume', default=False)
parser.add_argument('--tensorboard_dir', default="tensorboard")
parser.add_argument('--training_dir', default="training")
parser.add_argument('--resume_ckpt')


args = parser.parse_args()

def main(args):
    LOCAL_RANK = args.local_rank
    DIST = args.distributed
    BASE_CKPT_PATH = "/home/vp.shivasan/IvT/SparseAttentionViT/checkpoints/"

    if DIST:
        print("Intiliasing distributed process")
        DEVICE = torch.device('cuda:%d' % LOCAL_RANK)
        torch.cuda.set_device(LOCAL_RANK)
        print(DEVICE)
        dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.num_gpus, rank=LOCAL_RANK)
        print("Distributed process running")
    
    else:
        CUDA_ = 'cuda:1'
        DEVICE = torch.device(CUDA_)


    # BATCH_SIZE_TRAIN = 64
    # BATCH_SIZE_TEST = 128
    # IMAGE_SIZE = 368
    transform_train = torchvision.transforms.Compose([torchvision.transforms.Resize((args.image_size,args.image_size)),
                                                        transforms.RandomResizedCrop(args.image_size),
                                                        transforms.RandomHorizontalFlip(),
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

    transform_test = torchvision.transforms.Compose([torchvision.transforms.Resize((args.image_size,args.image_size)),
                                                        transforms.CenterCrop(args.image_size),
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
    train_set = torchvision.datasets.ImageFolder(root="/home/vp.shivasan/IvT/data/imagenette2/train",transform=transform_train)
    test_set = torchvision.datasets.ImageFolder(root="/home/vp.shivasan/IvT/data/imagenette2/val",transform=transform_test)
    

    if(DIST):
        sampler_train = torch.utils.data.distributed.DistributedSampler(train_set)
        sampler_val = torch.utils.data.distributed.DistributedSampler(test_set,shuffle = False)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_set)
        sampler_val = torch.utils.data.SequentialSampler(test_set)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.train_batch_size, drop_last=True)
    batch_sampler_val = torch.utils.data.BatchSampler(
        sampler_val, args.test_batch_size, drop_last=True)

    train_loader = torch.utils.data.DataLoader(train_set, batch_sampler=batch_sampler_train,num_workers=2)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_sampler=batch_sampler_val,num_workers=2)

    # MODEL_NAME = "vit_original"
    LOG_NAME = str(args.image_size)+ "_p" +str(args.patch_size)
    # LOG_NAME = "Test"
    print("LOG name: ",LOG_NAME)

    if not os.path.exists(BASE_CKPT_PATH+args.vit_arch):
        print("Creating model checkpoint dir")
        os.makedirs(BASE_CKPT_PATH+args.vit_arch)
    else:
        print("Model checkpoint dir exsist, make sure log name is different or else it will get overwritten")

    if(args.vit_arch == "OriginalViT"):
        model = ViT(
            image_size = args.image_size,
            patch_size = 16,
            num_classes = 10,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

    if(args.vit_arch == "BigBirdViT"):
        model = BigBirdViT(
            image_size = args.image_size,
            patch_size = 16,
            num_classes = 10,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

    # lr = 3e-5
    # gamma = 0.7
    # seed = 42

    # GPU ARGS
    # DIST = True
    # LOCAL_RANK = 0
    # NUM_GPUS = 4
    

    torch.manual_seed(317)
    torch.backends.cudnn.benchmark = True 
    
    # device = "cuda:2"

    model = model.to(DEVICE)
    if DIST:
        model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[LOCAL_RANK])                       
    else:
        model = nn.DataParallel(model,device_ids=[1,]).to('cuda:1')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    print("Training ", args.vit_arch)
    for epoch in tqdm(range(args.epochs)):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in train_loader:
            data = data.to(DEVICE)
            label = label.to(DEVICE)

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
                data = data.to(DEVICE)
                label = label.to(DEVICE)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)
        if(LOCAL_RANK == 1):
            print(
                f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
            )
    if(LOCAL_RANK == 1):
        torch.save(model.state_dict(), BASE_CKPT_PATH + args.vit_arch +"/" + LOG_NAME + ".pt")

if __name__ == '__main__':
  main(args)

# CLassic: python -m torch.distributed.launch --nproc_per_node=4 --node_rank=0 train_vit.py
# BigBird: python -m torch.distributed.launch --nproc_per_node=4 --node_rank=0 train_vit.py --vit_arch BigBirdViT

