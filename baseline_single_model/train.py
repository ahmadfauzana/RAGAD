import os
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from model import Encoder, Decoder  # Ensure this import works correctly
from dataset import get_data_transforms, MVTecDataset
from retrieval import find_similar_images
from detection import detect_anomalies
from utils import setup_seed, loss_function, save_checkpoint
import pdb

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.backends.cudnn.benchmark = True

def train(_class_, root='./mvtec/', ckpt_path='./ckpt/', ifgeom=None, tensorboard_log_dir='./runs/model/'):
    epochs = 50
    image_size = 256
    mode = "sp"  # 'sp' for single patch, 'mp' for multiple patches
    gamma = 1

    print("Training on ", _class_, " started")
    
    # Load pre-computed reference features 
    reference_features = np.load("D://Fauzan//Study PhD//Research//RAGAD//baseline_single_model//db_features//db_features_" + _class_ + ".npy").astype(np.float32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vq = mode == "sp"

    # Define data transformations for training and testing
    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = os.path.join(root, _class_, 'train')
    test_path = os.path.join(root, _class_)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    ckp_path = os.path.join(ckpt_path, f'wres50_{_class_}_I.pth' if mode == "sp" else f'wres50_{_class_}_P.pth')

    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")

    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)  # Reduced batch size
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False)

    encoder = Encoder(
        ch=128,
        out_ch=3,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        in_channels=3,
        resolution=image_size,
        z_channels=4,
        double_z=True,
        attn_type="vanilla"
    )

    decoder = Decoder(
        ch=128,
        out_ch=3,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        in_channels=3,
        resolution=image_size,
        z_channels=4,
        double_z=True,
        attn_type="vanilla"
    )

    encoder, decoder = map(lambda x: x.to(device), [encoder, decoder])
    
    optimizer = AdamW(list(decoder.parameters()), lr=0.005, betas=(0.5, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        for images in train_dataloader:
            inputs, _ = images
            inputs = inputs.to(device)
            
            # pdb.set_trace()
            features = encoder(inputs)  # Batch x C x H x W
            # print(features.shape)

            B, C, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(B * H * W, C)  # Reshape features

            similar_indices = [find_similar_images(f, reference_features) for f in features.detach().cpu().numpy()]
            retrieved_features = reference_features[similar_indices]
            retrieved_features = torch.tensor(retrieved_features, device=device).reshape(B, H, W, C).permute(0, 3, 1, 2)
            
            del features
            torch.cuda.empty_cache()
            
            outputs = decoder(retrieved_features)
            loss = loss_function(outputs, inputs)

            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            scores, _ = detect_anomalies(test_dataloader, reference_features, encoder, decoder, device, epoch)
            save_checkpoint(encoder, decoder, ckpt_path, epoch + 1)
            print(f'AUROC Score = {scores}')


if __name__ == '__main__':
    root_path = "D:\\Fauzan\\Study PhD\\Research\\DMAD\\mvtec_anomaly_detection\\"
    ckpt_path = "D:\\Fauzan\\Study PhD\\Research\\DMAD\\dataset\\ckpt\\ppdm\\"
    setup_seed(111)
    ifgeom = ['screw', 'carpet', 'metal_nut']
    item_list = ['capsule', 'cable','screw','pill','carpet', 'bottle', 'hazelnut','leather', 'grid','transistor', 'metal_nut', 'toothbrush', 'zipper', 'tile', 'wood']
    for i in item_list:
        start = time.time()
        train(i, root_path, ckpt_path, ifgeom=i in ifgeom)
        end = time.time()  # Record the end time of the entire training
        
        total = end - start  # Calculate the total training time
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)

        print('Total Training Time: {} hours {} minutes {} seconds'.format(int(hours), int(minutes), int(seconds)))
