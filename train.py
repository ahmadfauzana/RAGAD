import os
import cv2
import time
import torch
import random
import numpy as np
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from test import evaluation, visualize_component
from dataset import MVTecDataset
from resnet import wide_resnet50_2
from torch.nn import functional as F
from dataset import get_data_transforms
from de_resnet import de_wide_resnet50_2
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

ifgeom = ['screw', 'zipper', 'grid', 'dot', 'crack', 'scratch', 'hole', 'wood', 'tile', 'metal']

def setup_tensorboard(log_dir):
    writer = SummaryWriter(log_dir)
    return writer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_concat(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a.[0].shape[-1]
    for i in range(len(a)):
        a_map.append(F.interpolate(a[i], size=(size, size), mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[i], size=(size, size), mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map, dim=1)
    b_map = torch.cat(b_map, dim=1)
    loss += torch.mean(1 - cos_loss(a_map, b_map))
    return loss

def train(_class_, root='./mvtec/', ckpt_path='./checkpoints/', ifgeom=None, log_dir='./logs/'):
    print(f'Class: {_class_}')
    # Hyperparameters
    seed = 42
    epochs = 100
    image_size = 256
    mode = "sp"
    gamma = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(seed)
    writer = setup_tensorboard(log_dir)
    vq = mode == "sp"

    # Load dataset
    train_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = os.path.join(root, _class_, 'train')
    test_path = os.path.join(root, _class_, 'test')
    if not os.path.exists(train_path):
        train_path = os.path.join(root, _class_, 'train_good')
    if not os.path.exists(test_path):
        test_path = os.path.join(root, _class_, 'test_good')
    ckp_path = ckpt_path + 'wres50_' + _class_ + ('_I.pth' if mode == 'sp' else '_P.pth')
    train_data = ImageFolder(root=train_path, transform=train_transform)
    test_data = MVTecDataset(root=test_path, transform=train_transform, gt_transform=gt_transform, phase='test')
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    # Load model
    encoder, bn, offset = wide_resnet50_2(pretrained=True, vq=vq, gamma=gamma)
    encoder = encoder.to(device)
    bn = bn.to(device)
    offset = offset.to(device)
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    print(f'Encoder: {count_parameters(encoder)}')
    print(f'Decoder: {count_parameters(decoder)}')
    print(f'Offset: {count_parameters(offset)}')
    print(f'BatchNorm: {count_parameters(bn)}')
    print(f'VQ: {count_parameters(bn)}')
    
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(offset.parameters()) + list(bn.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Train
    print(f'Training {_class_}...')
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        offset.train()
        bn.train()
        train_loss = 0
        for i, (x, _) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            x = x.to(device)
            optimizer.zero_grad()
            if vq:
                z, _, _, _ = encoder(x)
                x_hat = decoder(z)
                loss = F.mse_loss(x_hat, x)
            else:
                z, _, _, _ = encoder(x)
                x_hat = decoder(z)
                loss = F.mse_loss(x_hat, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        writer.add_scalar('Loss/train', train_loss, epoch)
        print(f'Epoch: {epoch+1}/{epochs}, Loss: {train_loss:.4f}')
        scheduler.step()
        if (epoch+1) % 10 == 0:
            torch.save(encoder.state_dict(), ckp_path)
            print(f'Saved model at epoch {epoch+1}')
    print(f'Training {_class_} finished')
    torch.save(encoder.state_dict(), ckp_path)
    print(f'Saved model at epoch {epochs}')
    writer.close()
    print(f'Closed tensorboard')
    print(f'Evaluation...')
    encoder.load_state_dict(torch.load(ckp_path))
    encoder.eval()
    decoder.eval()
    offset.eval()
    bn.eval()
    evaluation(encoder, decoder, offset, bn, test_dataloader, device, _class_, ifgeom=ifgeom)
    print(f'Evaluation finished')
    print(f'Visualization...')
    visualize_component(encoder, decoder, offset, bn, test_dataloader, device, _class_, ifgeom=ifgeom)
    print(f'Visualization finished')
    print(f'Finished {_class_}')

if __name__ == '__main__':
    for _class_ in ifgeom:
        train(_class_)
    print('All finished')
    
