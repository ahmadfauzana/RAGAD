import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from dataset import get_data_transforms
from torchvision import transforms, datasets
from model import Encoder
import numpy as np
import pdb

# setting
image_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# define path folder and classes
root = "D://Fauzan//Study PhD//Research//DMAD//mvtec_anomaly_detection//"
save_path = "D://Fauzan//Study PhD//Research//RAGAD//baseline_single_model//db_features//"
item_list = ['capsule', 'cable','screw','pill','carpet', 'bottle', 'hazelnut','leather', 'grid','transistor', 'metal_nut', 'toothbrush', 'zipper', 'tile', 'wood']

# preprocessing
data_transform, gt_transform = get_data_transforms(image_size, image_size)  # Set the encoder to evaluation mode

encoder = Encoder(
    ch=128, 
    out_ch=3, 
    ch_mult=(1, 2, 4, 4), 
    num_res_blocks=2, 
    attn_resolutions=[ ],
    dropout=0.0, 
    in_channels=3, 
    resolution=image_size,
    z_channels=4, 
    double_z=True, 
    attn_type="vanilla"
)
encoder = encoder.to(device)
encoder.eval()

for item in item_list:
    # Create a list to store features
    features_list = []

    # Initialize the encoder

    root_path = os.path.join(root, item, 'train')
    dataset = ImageFolder(root=root_path, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Extract features for each image in the dataset
    with torch.no_grad():
        for inputs, _ in dataloader:
            # pdb.set_trace()
            
            inputs = inputs.to(device)
            features = encoder(inputs)  # Extract features from the last relevant layer

            # features B x C x H x W -> B x H x W x C -> BHW x C 
            B,C,H,W = features.shape

            features = features.permute(0,2,3,1) # B x H x W x C
            features = features.reshape(B*H*W, C) # BHW x C
            features_list.append(features.cpu().numpy())

    # Convert the list of features to a numpy array
    features_db = np.vstack(features_list)

    # Save features to a .npy file
    np.save(save_path + 'db_features_' + item + '.npy', features_db)
    print("Features saved to db_features_" + item + ".npy")