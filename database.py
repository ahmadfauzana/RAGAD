import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pandas as pd
import os
import numpy as np
from diffusers import AutoencoderKL

# Load the pre-trained AutoencoderKL model
model = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").cuda()

# Define preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 1: Preprocessing Data (Example: Read images from a folder)
data_folder = "D:\\Fauzan\\Study PhD\\Research\\DMAD\\mvtec_anomaly_detection"
image_files = os.listdir(data_folder)

# Get list of main folders
main_folders = [folder for folder in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, folder))]
print(f"Main folders: {main_folders}")

# Load or initialize the stable diffusion encoder model
# model = Encoder().cuda()  # Move model to GPU

# Encode images and store feature vectors in a dataframe
feature_vectors = []

# Iterate through main folders and retrieve images from the first subfolder in each
for main_folder in main_folders:
    subfolders = [subfolder for subfolder in os.listdir(os.path.join(data_folder, main_folder)) if os.path.isdir(os.path.join(data_folder, main_folder, subfolder))]
    print(f"Subfolders in {main_folder}: {subfolders}")
    if subfolders:
        subfolder = "train" if "train" in subfolders else subfolders[0]
        print(f"Reading images from {subfolder} in {main_folder}")
        images_in_subfolder = os.path.join(data_folder, main_folder, subfolder)
        dataset = ImageFolder(root=images_in_subfolder, transform=transform)

        for i, (image, label) in enumerate(dataset):
            # Convert image to tensor and move to GPU
            image_tensor = torch.tensor(image).float().cuda()

            # Add batch dimension if it doesn't exist
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)

            # Encode image using the stable diffusion encoder
            feature_vector = model.encode(image_tensor).latent_dist.mean
            feature_vectors.append({'image_path': dataset.imgs[i][0], 'feature_vector': feature_vector.cpu().detach()})  # Move back to CPU before appending

encoded_vectors = np.array(feature_vectors)
# Create a dataframe from feature vectors
df = pd.DataFrame(feature_vectors)

# Save dataframe to CSV file
np.save('feature_vectors.npy', encoded_vectors)
