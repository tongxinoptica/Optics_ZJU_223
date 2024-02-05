from unit import to_ssim_skimage
import torch
from torchvision import transforms
from PIL import Image
# from scipy.stats import pearsonr
import numpy as np
from skimage.metrics import structural_similarity as ssim

Tensor = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
Image_path = 'D:/train/input_data/all/419.jpg'
label_path = 'D:/train/groundtruth_data/all/419.jpg'
img = Image.open(Image_path)
img = Tensor(img)
img = torch.reshape(img, (1, 1, 256, 256)).squeeze(1)
label = Image.open(label_path)
label = Tensor(label)
label = torch.reshape(label, (1, 1, 256, 256)).squeeze(1)  # 增加batch维度


# def to_ssim_loss(img1, img2):
#     batch_img, H_img, W_img = img1.shape[0], img1.shape[1], img1.shape[2]
#     ssim_list = []
#     for b in range(batch_img):
#         img1_mean = torch.mean(img1[b])
#         img2_mean = torch.mean(img2[b])
#         img1_2 = torch.sum((img1[b] - img1_mean) ** 2) / (H_img * W_img - 1)
#         img2_2 = torch.sum((img2[b] - img2_mean) ** 2) / (H_img * W_img - 1)
#         img1_img2 = torch.sum((img1[b] - img1_mean) * (img2[b] - img2_mean)) / (H_img * W_img - 1)
#         ssim_list.append((2 * img1_mean * img2_mean + 0.0001) * (2 * img1_img2 + 0.0009) / (
#                 (img1_mean ** 2 + img2_mean ** 2 + 0.0001) * (img1_2 + img2_2 + 0.0009)))
#     ssim = sum(ssim_list) / len(ssim_list)
#     return ssim




a = to_ssim_loss(img, img)
