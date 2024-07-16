import os
import time
import faiss
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.cuda.amp import autocast, GradScaler
from model import Encoder, Decoder  # Assume models.py contains Encoder and Decoder definitions
from dataset import get_data_transforms, MVTecDataset
from retrieval import retrieve_similar_images
from detection import detect_anomalies
from utils import setup_seed 

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.backends.cudnn.benchmark = True

def train(_class_, root='./mvtec/', ckpt_path='./ckpt/', ifgeom=None, tensorboard_log_dir='./runs/model/'):
    epochs = 50
    image_size = 256
    mode = "sp"  # 'sp' for single patch, 'mp' for multiple patches
    gamma = 1

    # Load pre-computed reference features 
    reference_features = np.load("D:\\Fauzan\\Study PhD\\Research\\RAGAD\\db_features.npy").astype(np.float32)
    
    # Define the index
    dimension = reference_features.shape[1]
    nlist = 100  # Number of clusters
    m = 8  # Number of bytes per vector
    index = faiss.IndexIVFPQ(faiss.IndexFlatL2(dimension), dimension, nlist, m, 8)
    
    # Train the index
    index.train(reference_features)
    index.add(reference_features)

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

    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)  # Reduced batch size
    test_dataloader = DataLoader(test_data, batch_size=8, shuffle=False)

    encoder = Encoder(
        ch=64, 
        out_ch=128, 
        ch_mult=(1, 2, 4, 8), 
        num_res_blocks=2, 
        attn_resolutions=[16, 32, 64],
        dropout=0.1, 
        resamp_with_conv=True, 
        in_channels=3, 
        resolution=image_size,
        z_channels=image_size, 
        double_z=True, 
        use_linear_attn=False, 
        attn_type="vanilla"
    )
    decoder = Decoder(
        ch=64, 
        ch_mult=(1, 2, 4, 8), 
        num_res_blocks=2, 
        attn_resolutions=[16, 32, 64], 
        dropout=0.1, 
        resamp_with_conv=True, 
        in_channels=3,  
        resolution=256,  
        z_channels=256, 
        double_z=True  
    )

    encoder, decoder = map(lambda x: x.to(device), [encoder, decoder])
    
    optimizer = AdamW(list(decoder.parameters()), lr=0.005, betas=(0.5, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler() if torch.cuda.is_available() else None  # Use GradScaler only if CUDA is available

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        for images in train_dataloader:
            torch.cuda.empty_cache()
            inputs, _ = images
            inputs = inputs.to(device)

            if torch.cuda.is_available():
                with autocast():  # Mixed precision context
                    features = encoder(inputs)  # Batch x C x H x W

                    B, C, H, W = features.shape
                    features = features.permute(0, 2, 3, 1).reshape(B * H * W, C)  # Reshape features

                    similar_indices = [retrieve_similar_images(f, index) for f in features.detach().cpu().numpy()]
                    retrieved_features = reference_features[similar_indices]
                    retrieved_features = torch.tensor(retrieved_features, device=device).reshape(B, H, W, C).permute(0, 3, 1, 2)

                    optimizer.zero_grad()
                    outputs = decoder(retrieved_features)
                    loss = F.mse_loss(outputs, inputs)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                features = encoder(inputs)  # Batch x C x H x W

                B, C, H, W = features.shape
                features = features.permute(0, 2, 3, 1).reshape(B * H * W, C)  # Reshape features

                similar_indices = [retrieve_similar_images(f, index) for f in features.detach().cpu().numpy()]
                retrieved_features = reference_features[similar_indices]
                retrieved_features = torch.tensor(retrieved_features, device=device).reshape(B, H, W, C).permute(0, 3, 1, 2)

                optimizer.zero_grad()
                outputs = decoder(retrieved_features)
                loss = F.mse_loss(outputs, inputs)

                loss.backward()
                optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            encoder.eval()
            decoder.eval()
            scores = detect_anomalies(test_dataloader, reference_features, index, encoder, decoder, device)
            print(f'Anomaly Scores = {scores}')

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
