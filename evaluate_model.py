# Name: Bangcheng Wang
# time:
# evaluate_model.py

import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def evaluate_model(model, dataloader, device):
    model.to(device)
    model.eval()
    psnr_list = []
    ssim_list = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, inputs)  # 同上
            outputs = outputs.cpu().numpy()
            targets = targets.cpu().numpy()

            for output, target in zip(outputs, targets):
                psnr = peak_signal_noise_ratio(target, output)
                ssim = structural_similarity(target, output, multichannel=True)
                psnr_list.append(psnr)
                ssim_list.append(ssim)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    return avg_psnr, avg_ssim
