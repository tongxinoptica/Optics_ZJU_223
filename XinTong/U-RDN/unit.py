import torch
import torch.nn.functional as F
from math import log10
import torchvision.utils as utils
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
import numpy as np


# def to_psnr(img1, img2):
#     mse = F.mse_loss(img1, img2, reduction='none')
#     mse_split = torch.split(mse, 1)
#     mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
#
#     intensity_max = 1.0
#     psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
#     return psnr_list

def to_psnr(img1, img2):
    batch_img, channel_img, H_img, W_img = img1.shape[0], img1.shape[1], img1.shape[2], img1.shape[3]
    mse_list = []
    for b in range(batch_img):
        mse = torch.sum((img1[b] - img2[b]) ** 2) / (H_img * W_img)
        mse_list.append(mse)
    mse = sum(mse_list)/len(mse_list)
    psnr = 10.0 * log10(1 / mse)
    return psnr


def to_ssim_skimage(img1, img2):
    img1_list = torch.split(img1, 1)
    img2_list = torch.split(img2, 1)

    img1_list_np = [img1_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(img1_list))]
    img2_list_np = [img2_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(img2_list))]
    ssim_list = [ssim(img1_list_np[ind], img2_list_np[ind], data_range=1, multichannel=True) for ind in
                 range(len(img1_list))]
    return ssim_list


def to_ssim(img1, img2):
    batch_img, channel_img, H_img, W_img = img1.shape[0], img1.shape[1], img1.shape[2], img1.shape[3]
    ssim_list = []
    for b in range(batch_img):
        img1_mean = torch.mean(img1[b])
        img2_mean = torch.mean(img2[b])
        img1_2 = torch.sum((img1[b] - img1_mean) ** 2) / (H_img * W_img - 1)
        img2_2 = torch.sum((img2[b] - img2_mean) ** 2) / (H_img * W_img - 1)
        img1_img2 = torch.sum((img1[b] - img1_mean) * (img2[b] - img2_mean)) / (H_img * W_img - 1)
        ssim_list.append((2 * img1_mean * img2_mean + 0.0001) * (2 * img1_img2 + 0.0009) / (
                (img1_mean ** 2 + img2_mean ** 2 + 0.0001) * (img1_2 + img2_2 + 0.0009)))
    ssim_loss = sum(ssim_list) / len(ssim_list)
    return ssim_loss


def to_pearson(img1, img2):
    batch_img, channel_img, H_img, W_img = img1.shape[0], img1.shape[1], img1.shape[2], img1.shape[3]

    PCC_list = []
    for b in range(batch_img):
        img1_mean = torch.mean(img1[b])
        img2_mean = torch.mean(img2[b])
        y = img1[b] - img1_mean
        g = img2[b] - img2_mean
        yg = torch.sum(y * g)
        y2 = torch.sqrt(torch.sum(y * y))
        g2 = torch.sqrt(torch.sum(g * g))
        PCC_list.append(yg / (y2 * g2))
    pcc = -1.0*sum(PCC_list) / len(PCC_list)
    return pcc


def to_mseloss(img1, img2):
    batch_img, channel_img, H_img, W_img = img1.shape[0], img1.shape[1], img1.shape[2], img1.shape[3]
    mse_list = []
    for b in range(batch_img):
        mse = torch.sum((img1[b] - img2[b]) ** 2) / (H_img * W_img)
        mse_list.append(mse)
    mse = sum(mse_list) / len(mse_list)
    return mse
# def to_pearson(x, y):
#     x_bar = torch.mean(x, -1, True)
#     x_bar = torch.mean(x_bar, -2, True)
#     y_bar = torch.mean(y, -1, True)
#     y_bar = torch.mean(y_bar, -2, True)
#     a = torch.sum((x - x_bar) * (y - y_bar), -1)
#     a = torch.sum(a, -1)
#     b_1 = torch.sum((x - x_bar) ** 2, -1)
#     b_1 = torch.sum(b_1, -1)
#     b_2 = torch.sum((y - y_bar) ** 2, -1)
#     b_2 = torch.sum(b_2, -1)
#     b = torch.sqrt(b_1) * torch.sqrt(b_2)
#     npcc = -torch.mean(a / b)
#     return npcc

# transform = transforms.ToTensor()
# img_path = 'D:/val/input_data/1.jpg'
# img = Image.open(img_path)
# img = transform(img)
# # img = torch.reshape(img, (1, 1, 256, 256))
# label_path = 'D:/val/groundtruth_data/1.jpg'
# label = Image.open(label_path)
# label = transform(label)
# # label = torch.reshape(label, (1, 1, 256, 256))
# npcc = to_pearson(img,img)
# print(npcc)
# PSNR = to_psnr(img,label)
# SSIM = to_ssim_skimage(img, label)
# print(PSNR)
# print(SSIM)
