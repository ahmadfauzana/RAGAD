from torchvision import datasets, transforms
from PIL import Image
import os
import torch
import glob
import numpy as np
import random
import torch.utils.data as data
import torchvision.transforms.functional as TF
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder

def get_data_transforms(size, isize):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(isize),
        transforms.ToTensor(),
        transforms.Normalize(mean_train, std_train)
    ])
    gt_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(isize),
        transforms.ToTensor(),
        transforms.Normalize()
    ])
    return train_transform, gt_transform

class MVTecDataset(torch.utils.data.MVTecDataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == "train":
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transfrom = transform
        self.gt_transfrom = gt_transform
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()
    
    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []
        
        defect_types = os.listdir(self.img_path)
        for defect_type in defect_types:
            if defect_type == 'good':    
                img_paths = glob.glob(os.path.join(self.img_path, defect_type + '/*.png'))
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend[0] * len(img_paths)
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = sorted(glob.glob(os.path.join(self.img_path, defect_type + '/*.png')))
                img_tot_paths.extend(img_paths)
                gt_paths = sorted(glob.glob(os.path.join(self.gt_path, defect_type + '/*.png')))
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), 'The number of images and ground truth is not matched.'

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        gt_path = self.gt_paths[idx]
        label = self.labels[idx]
        defect_type = self.types[idx]

        img = Image.open(img_path).convert('RGB')
        
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt_path)
            gt = self.gt_transfrom(gt)
        
        assert img.size()[1:] == gt.size()[1:], 'The size of image and ground truth is not matched.'
        return img, gt, label, defect_type