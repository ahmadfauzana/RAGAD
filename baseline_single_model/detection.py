import torch
import numpy as np
from utils import loss_function
from retrieval import find_similar_images
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def compute_anomaly_score(original, reconstructed):
    return torch.mean((original - reconstructed) ** 2, dim=[1, 2, 3])

def compute_anomaly_map(original, reconstructed):
    # Calculate the absolute difference between original and reconstructed images
    anomaly_map = torch.abs(original - reconstructed)
    # Reduce to a single channel by taking the mean over the channel dimension
    anomaly_map = torch.mean(anomaly_map, dim=1, keepdim=True)
    return anomaly_map

def visualize_images(inputs, recon_image, anomaly_map, labels, save_path):
    inputs = inputs.cpu()
    recon_image = recon_image.cpu()
    anomaly_map = anomaly_map.cpu()
    labels = labels.cpu()

    fig, axs = plt.subplots(3, len(inputs), figsize=(15, 10))
    for i in range(len(inputs)):
        axs[0, i].imshow(inputs[i].permute(1, 2, 0))
        axs[0, i].set_title(f'Original {i} - Label: {labels[i].item()}')
        axs[0, i].axis('off')

        axs[1, i].imshow(anomaly_map[i, 0], cmap='hot')
        axs[1, i].set_title(f'Anomaly Map {i} - Label: {labels[i].item()}')
        axs[1, i].axis('off')

        axs[2, i].imshow(recon_image[i].permute(1, 2, 0))
        axs[2, i].set_title(f'Reconstructed {i} - Label: {labels[i].item()}')
        axs[2, i].axis('off')

    plt.savefig(save_path)
    plt.show()

# Detect anomalies
def detect_anomalies(dataloader, reference_features, encoder, decoder, device, number):
    encoder.eval()
    decoder.eval()
    anomaly_scores = []
    all_labels = []
    with torch.no_grad():
        for images in dataloader:
            torch.cuda.empty_cache()
            inputs, masks, labels, desc = images
            inputs = inputs.to(device)
            labels = labels.to(device)
            features = encoder(inputs)  # Batch x C x H x W
            
            B, C, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(B * H * W, C)  # Reshape features

            similar_indices = [find_similar_images(f, reference_features) for f in features.detach().cpu().numpy()]
            retrieved_features = reference_features[similar_indices]
            retrieved_features = torch.tensor(retrieved_features, device=device).reshape(B, H, W, C).permute(0, 3, 1, 2)

            recon_image = decoder(retrieved_features)

            anomaly_map = compute_anomaly_map(inputs, recon_image)
            anomaly_score = compute_anomaly_score(inputs, recon_image)
            anomaly_scores.extend(anomaly_score.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            visualize_images(inputs, recon_image, anomaly_map, labels, 'output/visualization_anomap_' + str(int(number / 10 + 1)) + '.png')
            visualize_images(inputs, recon_image, anomaly_score, labels, 'output/visualization_anoscore' + str(int(number / 10 + 1)) + '.png')
            
    # update on anomaly score with ground truth and reconstruction image
    anomaly_scores = np.array(anomaly_scores)

    # Compute AUC
    auc_score_anoscore = roc_auc_score(masks, anomaly_scores)
    auc_score_anomap = roc_auc_score(masks, anomaly_map)
    
    print(f"AUC Score: {auc_score_anoscore} related to Anomaly Score")
    print(f"AUC Score: {auc_score_anomap} related to Anomaly Map")

    return auc_score_anoscore, auc_score_anomap