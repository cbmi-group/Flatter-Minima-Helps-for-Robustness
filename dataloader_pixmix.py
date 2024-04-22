import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import pandas as pd

import random

import numpy as np
import os
import PIL
import torch
import torchvision

from PIL import Image
from torch.utils.data import Subset
from torchvision import datasets
import pixmix_utils as utils


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


class TinyImageNet(Dataset):
    def __init__(self, dataset_type, transform=None):
        self.root = "./data/tiny-imagenet-200/"
        data_path = os.path.join(self.root, dataset_type)

        self.dataset = torchvision.datasets.ImageFolder(root=data_path)

        self.transform = transform

    def __getitem__(self, index):
        img, targets = self.dataset[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, targets

    def __len__(self):
        return self.dataset.__len__()


class TinyImageNetC(Dataset):
    def __init__(self, name, data_dir="./data/Tiny-ImageNet-C", level=1):
        self.corruptions = [
            "gaussian_noise",
            "shot_noise",
            "speckle_noise",
            "impulse_noise",
            "defocus_blur",
            "gaussian_blur",
            "motion_blur",
            "zoom_blur",
            "snow",
            "fog",
            "brightness",
            "contrast",
            "elastic_transform",
            "pixelate",
            "jpeg_compression",
            "spatter",
            "saturate",
            "frost"
        ]

        assert name in self.corruptions
        self.root = data_dir
        data_path = os.path.join(self.root, name+"/"+str(level))

        self.dataset = torchvision.datasets.ImageFolder(root=data_path)
        # print(self.dataset.__len__())
        # print(len(self.dataset.classes))
        # print(self.dataset.classes)
        # print(self.dataset.class_to_idx)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, index):
        img, targets = self.dataset[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, targets

    def __len__(self):
        return self.dataset.__len__()


class CIFAR100C(Dataset):
    def __init__(self, name, data_dir="/mnt/data1/ZKJ_data/CIFAR-100-C"):
        self.corruptions = [
            "gaussian_noise",
            "shot_noise",
            "speckle_noise",
            "impulse_noise",
            "defocus_blur",
            "gaussian_blur",
            "glass_blur",
            "motion_blur",
            "zoom_blur",
            "snow",
            "fog",
            "brightness",
            "contrast",
            "elastic_transform",
            "pixelate",
            "jpeg_compression",
            "spatter",
            "saturate",
            "frost"
        ]

        assert name in self.corruptions
        self.root = data_dir
        data_path = os.path.join(self.root, name + '.npy')
        target_path = os.path.join(self.root, 'labels.npy')

        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, targets

    def __len__(self):
        return len(self.data)


class CIFAR10C(Dataset):
    def __init__(self, name, data_dir="./data/CIFAR-10-C"):
        self.corruptions = [
            "gaussian_noise",
            "shot_noise",
            "speckle_noise",
            "impulse_noise",
            "defocus_blur",
            "gaussian_blur",
            "glass_blur",
            "motion_blur",
            "zoom_blur",
            "snow",
            "fog",
            "brightness",
            "contrast",
            "elastic_transform",
            "pixelate",
            "jpeg_compression",
            "spatter",
            "saturate",
            "frost"
        ]

        assert name in self.corruptions
        self.root = data_dir
        data_path = os.path.join(self.root, name + '.npy')
        target_path = os.path.join(self.root, 'labels.npy')

        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, targets

    def __len__(self):
        return len(self.data)

def augment_input(image):
  aug_list = utils.augmentations_all
  op = np.random.choice(aug_list)
  return op(image.copy(), 3)

def pixmix(orig, mixing_pic, preprocess):
    mixings = utils.mixings
    tensorize, normalize = preprocess['tensorize'], preprocess['normalize']
    if np.random.random() < 0.5:
        mixed = tensorize(augment_input(orig))
    else:
        mixed = tensorize(orig)

    for _ in range(np.random.randint(4 + 1)):

        if np.random.random() < 0.5:
            aug_image_copy = tensorize(augment_input(orig))
        else:
            aug_image_copy = tensorize(mixing_pic)

        mixed_op = np.random.choice(mixings)
        mixed = mixed_op(mixed, aug_image_copy, 3)
        mixed = torch.clip(mixed, 0, 1)

    return normalize(mixed)

class PixMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform PixMix."""

  def __init__(self, dataset, mixing_set, preprocess):
    self.dataset = dataset
    self.mixing_set = mixing_set
    self.preprocess = preprocess

  def __getitem__(self, i):
    x, y = self.dataset[i]
    rnd_idx = np.random.choice(len(self.mixing_set))
    mixing_pic, _ = self.mixing_set[rnd_idx]
    return pixmix(x, mixing_pic, self.preprocess), y

  def __len__(self):
    return len(self.dataset)

def create_dataloader(dataset, batch_size, use_val=True, transform_dict=None):
    if dataset == "TinyImageNet":
        if transform_dict is not None:
            transform_train, transform_test = transform_dict["train"], transform_dict["test"]
        
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])



        train_dataset = TinyImageNet("train", transform_train)
        testset = TinyImageNet("val", transform_test)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        valloader = None
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)        

    if dataset == "CIFAR10":
        if transform_dict is not None:
            transform_train, transform_test = transform_dict["train"], transform_dict["test"]
        else:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ])

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])

            mixing_set_transform = transforms.Compose(
                [transforms.Resize(36),
                 transforms.RandomCrop(32)])
        
        train_dataset = datasets.CIFAR10(root='../cifar_data', train=True, download=True, transform=transform_train)
        mixing_set = datasets.ImageFolder('../pixmix_main/fractals', transform=mixing_set_transform)

        print('image sie:', len(train_dataset))
        print('aug_size', len(mixing_set))

        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        to_tensor = transforms.ToTensor()

        train_dataset = PixMixDataset(train_dataset, mixing_set,{'normalize': normalize, 'tensorize': to_tensor})

        if use_val:

            valid_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)

            train_idx = np.loadtxt("./data/train_idx.txt", dtype=int)
            valid_idx = np.loadtxt("./data/val_idx.txt", dtype=int)

            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            trainloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=True,
                num_workers=8, worker_init_fn=seed_worker, generator=g
            )
            valloader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=batch_size, sampler=valid_sampler,
                num_workers=8, worker_init_fn=seed_worker, generator=g
            )
        else:
            trainloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
            valloader = None
        testset = torchvision.datasets.CIFAR10(
            root='../cifar_data', train=False, download=True, transform=transform_test)

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=8)

    if dataset == "CIFAR100":
        if transform_dict is not None:
            transform_train, transform_test = transform_dict["train"], transform_dict["test"]
        else:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ])

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])

            mixing_set_transform = transforms.Compose(
                [transforms.Resize(36),
                 transforms.RandomCrop(32)])

        train_dataset = datasets.CIFAR100(root='../cifar_data', train=True, download=True, transform=transform_train)
        mixing_set = datasets.ImageFolder('../pixmix_main/fractals', transform=mixing_set_transform)

        print('image sie:', len(train_dataset))
        print('aug_size', len(mixing_set))

        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        to_tensor = transforms.ToTensor()

        train_dataset = PixMixDataset(train_dataset, mixing_set, {'normalize': normalize, 'tensorize': to_tensor})

        # load the dataset
        if use_val:

            valid_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_test, )

            indices = list(range(50000))
            np.random.shuffle(indices)

            train_idx, valid_idx = indices[:45000], indices[45000:]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            
            trainloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=True,
                num_workers=8, worker_init_fn=seed_worker, generator=g
            )
            valloader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=batch_size, sampler=valid_sampler,
                num_workers=8, worker_init_fn=seed_worker, generator=g
            )
        
        else:
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
            valloader = None

        testset = torchvision.datasets.CIFAR100(root='../cifar_data', train=False, download=True, transform=transform_test)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    return trainloader, valloader, testloader