import torch 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from torch.nn import functional as F

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        amap = np.ones([out_size, out_size])
    else:
        amap = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(fs_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=(out_size, out_size), mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            amap *= a_map
        else:
            amap += a_map
    return amap, a_map_list

def show_cam_on_image(img, anomaly_map):
    heatmap = cv2.applyColorMap(np.uint8(255 * anomaly_map), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image

def cv2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(255 * gray), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    return heatmap

def minmax(list):
    return (list - np.min(list)) / (np.max(list) - np.min(list))

def evaluation(offset, encoder, bn, decoder, dataloader, device, _class_=None, mode=None, ifgeom=None):
    offset.eval()
    encoder.eval()
    bn.eval()
    decoder.eval()
    with torch.no_grad():
        for i, (input, target, label) in enumerate(tqdm(dataloader)):
            input = input.to(device)
            target = target.to(device)
            label = label.to(device)
            fs_list, ft_list, _, _ = offset(input)
            amap, a_map_list = cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul')
            if mode == 'cam':
                for j in range(input.shape[0]):
                    img = input[j].to('cpu').detach().numpy().transpose(1, 2, 0)
                    img = min_max_norm(img)
                    cam = show_cam_on_image(img, amap)
                    plt.imshow(cam)
                    plt.show()
            elif mode == 'heatmap':
                for j in range(input.shape[0]):
                    heatmap = cv2heatmap(amap)
                    plt.imshow(heatmap)
                    plt.show()
            elif mode == 'anomaly_map':
                for j in range(input.shape[0]):
                    plt.imshow(amap)
                    plt.show()
            elif mode == 'all':
                for j in range(input.shape[0]):
                    img = input[j].to('cpu').detach().numpy().transpose(1, 2, 0)
                    img = min_max_norm(img)
                    cam = show_cam_on_image(img, amap)
                    heatmap = cv2heatmap(amap)
                    plt.imshow(cam)
                    plt.show()
                    plt.imshow(heatmap)
                    plt.show()
                    plt.imshow(amap)
                    plt.show()
            else:
                raise ValueError('Invalid mode')
            if i == 5:
                break