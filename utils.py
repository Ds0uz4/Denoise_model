import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def denormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img = img.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    return img.clamp(0, 1)

def psnr_ssim_evaluation(noisy_images, clean_images):
    psnr_scores = []
    ssim_scores = []

    for idx, (noisy, clean) in enumerate(zip(noisy_images, clean_images)):
        if torch.is_tensor(noisy):
            noisy = noisy.cpu().detach().numpy()
        if torch.is_tensor(clean):
            clean = clean.cpu().detach().numpy()

        if noisy.ndim == 3 and noisy.shape[0] in [1, 3]:
            noisy = np.transpose(noisy, (1, 2, 0))
        if clean.ndim == 3 and clean.shape[0] in [1, 3]:
            clean = np.transpose(clean, (1, 2, 0))

        noisy = noisy.astype(np.float32)
        clean = clean.astype(np.float32)

        if noisy.max() > 1.0:
            noisy = noisy / 255.0
        if clean.max() > 1.0:
            clean = clean / 255.0

        psnr_value = psnr(clean, noisy, data_range=1)
        
        if noisy.ndim == 2 or (noisy.ndim == 3 and noisy.shape[-1] == 1):
            ssim_value = ssim(clean.squeeze(), noisy.squeeze(), data_range=1)
        else:
            ssim_value = ssim(clean, noisy, data_range=1, channel_axis=-1)

        psnr_scores.append(psnr_value)
        ssim_scores.append(ssim_value)

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)

    return avg_psnr, avg_ssim

def normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img = img.clone()
    for c in range(3):
        img[c] = (img[c] - mean[c]) / std[c]
    return img