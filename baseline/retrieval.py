import faiss
import torch
import numpy as np
from scipy.spatial import KDTree
import pdb
from annoy import AnnoyIndex

class ImageRetrievalSystem:
    def __init__(self, autoencoder, latent_dim):
        self.autoencoder = autoencoder
        self.index = faiss.IndexFlatL2(latent_dim)

    def build_index(self, dataloader, features):
        self.autoencoder.eval()
        latent_vectors = []
        with torch.no_grad():
            for batch in dataloader:
                imgs = batch[0]
                h = self.autoencoder.encoder(imgs)
                mu, _ = torch.chunk(h, 2, dim=1)
                latent_vectors.append(mu.cpu().numpy())
        latent_vectors = np.vstack(latent_vectors)
        self.index.add(latent_vectors)

    def query(self, image):
        self.autoencoder.eval()
        with torch.no_grad():
            h = self.autoencoder.encoder(image.unsqueeze(0))
            mu, _ = torch.chunk(h, 2, dim=1)
        distances, indices = self.index.search(mu.cpu().numpy(), k=5)
        return distances, indices

# Retrieve similar images
def retrieve_similar_images(features, index):
    kdtree = KDTree(list(index.values()))
    similar_indices = []
    for feature in features:
        distances, indices = kdtree.query(feature, k=10)
        similar_indices.append(indices)
    return similar_indices, _