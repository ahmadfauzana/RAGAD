import html
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision import models, transforms


def encode_images(encoder, images):
    batch_features = encoder.encode(images).latent_dist.mean
    return batch_features

def load_database_features(feature_file_path):
    """
    Load database features from a .npy file.
    """
    return np.load(feature_file_path, allow_pickle=True)

# def retrieval_process(query_images):
#     print("Retrieval process started...")

#     feature_file_path = "D:\\Fauzan\\Study PhD\\Research\\RAGAD\\feature_vectors.npy"
#     # Load precomputed database features
#     database_features = load_database_features(feature_file_path)
#     # Pastikan untuk mengonversi setiap feature menjadi tensor PyTorch dari nilai array atau tensor, bukan dari dictionary
#     database_features = [torch.tensor(feature['feature_vector']) for feature in database_features if 'feature_vector' in feature]
#     for i, tensor in enumerate(database_features):
#         print(f"Database Features {i}: Size {tensor.size()}")

#     # For "inputs" input
#     # Determine the target size for resizing
    
#     # Resize tensors to have the same size
#     for i, tensor in enumerate(query_images):
#         print(f"Tensor {i}: Size {tensor.size()}")

#     # target_size = query_images[0].size(-1)
#     # db_features = []
#     # for db_feature in database_features:
#     #     if db_feature.dim() == 2:  # Hanya memiliki H dan W
#     #         db_feature = db_feature.unsqueeze(0)
#     #     db_feature_resized = F.interpolate(db_feature, size=(target_size, target_size), mode='bilinear', align_corners=False)
#     #     db_features.append(db_feature_resized)

#     # # print("DB Features: ", db_features[0])

#     # # Compute similarity between query images and images in the database
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     # query_tensor_resized = query_images[0].repeat(1, 1, 1, db_features[0].size(3))

#     # # Compute cosine similarity for each tensor in db_features
#     similarities = [torch.nn.functional.cosine_similarity(query_images, db_feature.to(device), dim=1) for db_feature in database_features]
#     # results = similarities[:3].tolist()

#     return similarities

def retrieval_process(query_images):
    print("Retrieval process started...")

    feature_file_path = "D:\\Fauzan\\Study PhD\\Research\\RAGAD\\feature_vectors.npy"
    # Load precomputed database features
    database_features = load_database_features(feature_file_path)
    # Convert each feature to PyTorch tensor from array or tensor values, not from dictionary
    database_features = [torch.tensor(feature['feature_vector']) for feature in database_features if 'feature_vector' in feature]
    for i, tensor in enumerate(database_features):
        print(f"Database Features {i}: Size {tensor.size()}")

    # Resize tensors to have the same size
    target_size = query_images[0].size(-1)
    db_features = []
    for db_feature in database_features:
        if db_feature.dim() == 2:  # Only has H and W
            db_feature = db_feature.unsqueeze(0)
        db_feature_resized = F.interpolate(db_feature, size=(target_size, target_size), mode='bilinear', align_corners=False)
        db_features.append(db_feature_resized)

    # Compute similarity between query images and images in the database
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Compute cosine similarity for each tensor in db_features
    similarities = [torch.nn.functional.cosine_similarity(query_images.to(device), db_feature.to(device), dim=1) for db_feature in db_features]

    return similarities

def loss_function(a, b):
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        #loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss