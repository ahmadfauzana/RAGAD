import os
import cv2
import time
import torch
import random
import numpy as np
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from test import evaluation
from dataset import MVTecDataset
from resnet import wide_resnet50_2
from torch.nn import functional as F
from dataset import get_data_transforms
from de_resnet import de_wide_resnet50_2
from torchvision.datasets import ImageFolder
from rdm.modules.custom.utils import retrieval_process
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

ifgeom = ['screw', 'zipper', 'grid', 'dot', 'crack', 'scratch', 'hole', 'wood', 'tile', 'metal']

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_function(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss

def loss_concat(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
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
    vq = mode == "sp"

    # Load dataset
    train_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = os.path.join(root, _class_, 'train')
    test_path = os.path.join(root, _class_)
    if not os.path.exists(train_path):
        train_path = os.path.join(root, _class_, 'train_good')
    if not os.path.exists(test_path):
        test_path = os.path.join(root, _class_, 'test_good')
    ckp_path = ckpt_path + 'wres50_' + _class_ + ('_I.pth' if mode == 'sp' else '_P.pth')
    train_data = ImageFolder(root=train_path, transform=train_transform)
    test_data = MVTecDataset(root=test_path, transform=train_transform, gt_transform=gt_transform, phase='test')
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    model_id = "stabilityai/stable-diffusion-2"
    
    # Load model
    encoder, bn, offset = wide_resnet50_2(pretrained=True, vq=vq, gamma=gamma)
    encoder = encoder.to(device)
    bn = bn.to(device)
    offset = offset.to(device)
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(offset.parameters()) + list(bn.parameters()), lr=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Load retriever
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    retriever = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    
    # Train
    print(f'Training {_class_}...')
    for epoch in range(epochs):
        offset.train()
        bn.train()
        decoder.train()
        loss_rec = {"main":[0],
                    "offset":[0],
                    "vq":[0]}
        for k, (img, label) in enumerate(train_dataloader):
            img = img.to(device)
            _, img_, offset_loss = offset(img)
            inputs = encoder(img_)
            vq, vq_loss = bn(inputs)
            # database features implemented in function below using stable diffusion encoder
            similar_features = retrieval_process(retriever, img, img_, _class_, 10)
            # swap during testing
            retrieval_features = torch.cat([*similar_features], dim=0)
            outputs = decoder(vq)
            main_loss = loss_function(retrieval_features, outputs)
            loss = main_loss + offset_loss + vq_loss   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_rec["main"].append(main_loss.item())
            loss_rec["offset"].append(offset_loss.item())
            try:
                loss_rec["vq"].append(vq_loss.item())
            except:
                loss_rec["vq"].append(0)
        print('epoch [{}/{}], main_loss:{:.4f}, offset_loss:{:.4f}, vq_loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_rec["main"]), np.mean(loss_rec["offset"]), np.mean(loss_rec["vq"])))
        if (epoch + 1) % 10 == 0:
            auroc = evaluation(offset, encoder, bn, decoder, test_dataloader, device, _class_, mode, ifgeom)
            print(f'Saved model at epoch {epoch+1}')
            torch.save({
                'offset': offset.state_dict(),
                'bn': bn.state_dict(),
                'decoder': decoder.state_dict()}, ckp_path)
            print('Auroc:{:.3f}'.format(auroc))
            
    print(f'Training {_class_} finished')
    torch.save(encoder.state_dict(), ckp_path)
    print(f'Saved model at epoch {epochs}')
    print(f'Evaluation...')
    encoder.load_state_dict(torch.load(ckp_path))
    encoder.eval()
    decoder.eval()
    offset.eval()
    bn.eval()
    evaluation(encoder, decoder, offset, bn, test_dataloader, device, _class_, ifgeom=ifgeom)
    print(f'Evaluation finished')
    print(f'Finished {_class_}')

    scheduler.step()

if __name__ == '__main__':
    root_path = "D:\\Fauzan\\Study PhD\\Research\\DMAD\\mvtec_anomaly_detection"
    ckpt_path = "D:\\Fauzan\\Study PhD\\Research\\DMAD\\dataset\\ckpt\\ppdm"
    item_list = ['screw', 'zipper', 'grid', 'dot', 'crack', 'scratch', 'hole', 'wood', 'tile', 'metal']
    for i in item_list:
        train(i, root=root_path, ckpt_path=ckpt_path, ifgeom=ifgeom, log_dir='./logs/')
    print('All finished')