import torch 
from dataset import get_data_transforms, MVTecDataset, load_data
import numpy as np
from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import random
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.utils as vutils
import cv2
import time

def cal_anomaly_map(model, data, device):
    model.eval()
    with torch.no_grad():
        data = data.to(device)