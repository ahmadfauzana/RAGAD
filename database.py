import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from diffusers import AutoencoderKL

class ImageFeatureExtractor:
    def __init__(self, data_folder, model_name="CompVis/stable-diffusion-v1-4", image_size=(224, 224)):
        self.data_folder = data_folder
        self.model = AutoencoderKL.from_pretrained(model_name, subfolder="vae").cuda()
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_main_folders(self):
        return [folder for folder in os.listdir(self.data_folder) if os.path.isdir(os.path.join(self.data_folder, folder))]

    def encode_images(self):
        latent_vectors = []
        image_paths = []
        main_folders = self.get_main_folders()
        print(f"Main folders: {main_folders}")

        for main_folder in main_folders:
            subfolders = [subfolder for subfolder in os.listdir(os.path.join(self.data_folder, main_folder)) if os.path.isdir(os.path.join(self.data_folder, main_folder, subfolder))]
            print(f"Subfolders in {main_folder}: {subfolders}")
            if subfolders:
                subfolder = "train" if "train" in subfolders else subfolders[0]
                print(f"Reading images from {subfolder} in {main_folder}")
                images_in_subfolder = os.path.join(self.data_folder, main_folder, subfolder)
                dataset = ImageFolder(root=images_in_subfolder, transform=self.transform)

                for i, (image, _) in enumerate(dataset):
                    image_tensor = image.unsqueeze(0).cuda()
                    with torch.no_grad():
                        latent_vector = self.model.encode(image_tensor)
                    latent_vectors.append(latent_vector.squeeze(0))
                    image_paths.append(images_in_subfolder)

        return latent_vectors, image_paths

    # def save_feature_vectors(self, feature_vectors, filename='feature_vectors.npy'):
    #     np.save(filename, feature_vectors)
    #     print(f"Feature vectors saved to {filename}")

# if __name__ == "__main__":
#     data_folder = "D:\\Fauzan\\Study PhD\\Research\\DMAD\\mvtec_anomaly_detection"  # Use relative path
#     extractor = ImageFeatureExtractor(data_folder)
#     feature_vectors = extractor.encode_images()
#     df = pd.DataFrame(feature_vectors)
#     extractor.save_feature_vectors(df.to_dict('records'))
