import torch
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_tensorboard(log_dir):
    return SummaryWriter(log_dir)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def loss_function(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = sum(0.1 * mse_loss(a_item, b_item) + torch.mean(1 - cos_loss(a_item.view(a_item.shape[0], -1), b_item.view(b_item.shape[0], -1))) for a_item, b_item in zip(a, b))
    return loss

def loss_concat(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    a_map = [F.interpolate(a_item, size=a[0].shape[-1], mode='bilinear', align_corners=True) for a_item in a]
    b_map = [F.interpolate(b_item, size=b[0].shape[-1], mode='bilinear', align_corners=True) for b_item in b]
    a_map = torch.cat(a_map, 1)
    b_map = torch.cat(b_map, 1)
    loss = torch.mean(1 - cos_loss(a_map, b_map))
    return loss

def save_checkpoint(encoder, decoder, checkpoint_path, batch_number):
    checkpoint = {
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'batch_number': batch_number
    }
    torch.save(checkpoint, f"{checkpoint_path}/checkpoint_{batch_number}.pt")

def load_checkpoint(encoder, decoder, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    return checkpoint['batch_number']