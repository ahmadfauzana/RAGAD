import os
import torch
import random
from PIL import Image
import numpy as np
from database import ImageFeatureExtractor
from torchvision import transforms
from dataset import get_data_transforms, MVTecDataset
from torchvision.datasets import ImageFolder
from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2
from test import evaluation
from torch.nn import functional as F
from utils import loss_function
from model_retrieval import retrieve_similar_images, generate_images_with_stable_diffusion, get_prompts_for_images

def setup_environment():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def setup_model(device):
    encoder, bn, offset = wide_resnet50_2(pretrained=True, vq=False, gamma=1)
    decoder = de_wide_resnet50_2(pretrained=False)
    encoder = encoder.to(device)
    bn = bn.to(device)
    offset = offset.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    return encoder, bn, offset, decoder

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# Train the retrieval augmented generation system
def train(device, _class_, root='./mvtec/', ckpt_path='./ckpt/', ifgeom=None):
    print(f"Training {_class_} on {device}")
    
    setup_environment()
    setup_seed(111)

    epochs = 400
    learning_rate = 0.005
    batch_size = 8
    image_size = 256
    mode = "sp"

    # Laod caching database
    features_db = np.load('features_db.npy')

    # Paths
    train_path = os.path.join(root, _class_, 'train')
    test_path = os.path.join(root, _class_)
    ckp_path = os.path.join(ckpt_path, f'wres50_{_class_}_{"I" if mode == "sp" else "P"}.pth')

    # Data
    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    # Load the pre-trained encoder, bottleneck, and offset networks
    encoder, bn, offset, decoder = setup_model(device)

    # Set up the optimizer and scheduler
    optimizer = torch.optim.AdamW(list(offset.parameters()) + list(decoder.parameters()) + list(bn.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        offset.train()
        bn.train()
        decoder.train()
        loss_rec = {"main": [0], "offset": [0], "vq": [0]}
        for k, (img, label) in enumerate(train_dataloader):
            img = img.to(device)
            _, img_, offset_loss = offset(img)
            inputs = encoder(img_)[-1]
            features = inputs.view(features.size(0), -1).cpu().numpy()

            # switching using  external db
            gen_images = []
            for feature in features:
                similar_image_indices = retrieve_similar_images(feature)
                prompts = get_prompts_for_images(similar_image_indices)
                generated_images = generate_images_with_stable_diffusion(prompts)
                gen_images.append(generated_images)

            vq, vq_loss = bn(gen_images)
            outputs = decoder(vq)
            main_loss = loss_function(inputs, outputs)
            loss = main_loss + offset_loss + vq_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_rec["main"].append(main_loss.item())
            loss_rec["offset"].append(offset_loss.item())
            try:
                loss_rec["vq"].append(vq_loss.item())
            except:
                loss_rec["vq"].append(0)
        print(f'epoch [{epoch + 1}/{epochs}], main_loss:{np.mean(loss_rec["main"]):.4f}, offset_loss:{np.mean(loss_rec["offset"]):.4f}, vq_loss:{np.mean(loss_rec["vq"]):.4f}')
        if (epoch + 1) % 10 == 0:
            auroc = evaluation(offset, encoder, bn, decoder, test_dataloader, device, _class_, mode, ifgeom)
            torch.save({
                'offset': offset.state_dict(),
                'bn': bn.state_dict(),
                'decoder': decoder.state_dict()}, ckp_path)
            print(f'Auroc:{auroc:.3f}')
        scheduler.step()

if __name__ == '__main__':
    root_path = "D:\\Fauzan\\Study PhD\\Research\\DMAD\\mvtec_anomaly_detection\\"
    ckpt_path = "D:\\Fauzan\\Study PhD\\Research\\DMAD\\dataset\\ckpt\\ppdm\\"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # item_list = ['capsule', 'cable', 'screw', 'pill', 'carpet', 'bottle', 'hazelnut', 'leather', 'grid', 'transistor', 'metal_nut', 'toothbrush', 'zipper', 'tile', 'wood']
    # for i in item_list:
    train(device, "cable", root_path, ckpt_path, ifgeom=i in ['screw', 'carpet', 'metal_nut'])