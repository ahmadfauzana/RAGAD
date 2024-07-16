import os
import torch
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from dataset import get_data_transforms
from resnet import wide_resnet50_2

# setting
image_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# define path folder and classes
root = "D:\\Fauzan\\Study PhD\\Research\\DMAD\\mvtec_anomaly_detection\\"
item_list = ['capsule', 'cable', 'screw', 'pill', 'carpet', 'bottle', 'hazelnut', 'leather', 'grid', 'transistor', 'metal_nut', 'toothbrush', 'zipper', 'tile', 'wood']

# preprocessing
data_transform, gt_transform = get_data_transforms(image_size, image_size)

# Create a matrix to store the features
features_db = []

# Initialize the ResNet model
model, bn, offset = wide_resnet50_2(pretrained=True)
model = model.to(device)
model.eval()

# Load dataset
for item in item_list:
    root_path = os.path.join(root, item, 'train')
    dataset = ImageFolder(root=root_path, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Extract features for each image in the dataset
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            features = model(inputs)[-1]  # Extract features from the last relevant layer
            features = features.view(features.size(0), -1)  # Flatten the features
            features_db.append(features.cpu().numpy())

# Convert the list of features to a numpy array
features_db = np.vstack(features_db)

# Save the features to a file
np.save('features_db.npy', features_db)