import torch
import numpy as np
from utils import loss_function
from retrieval import retrieve_similar_images

def compute_anomaly_score(original, reconstructed):
    return torch.mean((original - reconstructed) ** 2, dim=[1, 2, 3])

def anomaly_score(model, image, criterion):
    image = image.view(1, -1)
    with torch.no_grad():
        reconstruction = model(image)
    error = loss_function(reconstruction, image)
    return error.item()

# Detect anomalies
def detect_anomalies(dataloader, reference_features, index, encoder, autoencoder, device):
    autoencoder.eval()
    anomaly_scores = []
    with torch.no_grad():
        for image in dataloader:
            image = image.to(device)
            feature = encoder(image).squeeze()
            similar_indices = retrieve_similar_images(feature.cpu().numpy(), index)
            retrieved_features = reference_features[similar_indices]
            input_features = torch.cat([feature, torch.tensor(retrieved_features, device=device)], dim=0)

            recon_image, _, _ = autoencoder(input_features)
            anomaly_score = compute_anomaly_score(image.view(-1, 224 * 224 * 3), recon_image)
            anomaly_scores.append(anomaly_score.item())

    return np.array(anomaly_scores)