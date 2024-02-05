import torch
import torch.nn as nn
from unit import to_pearson, to_ssim, to_mseloss


class NPCC_loss(nn.Module):
    def __init__(self):
        super(NPCC_loss, self).__init__()

    def forward(self, img1, img2):
        return to_pearson(img1, img2)


class ssim_loss(nn.Module):
    def __init__(self):
        super(ssim_loss, self).__init__()

    def forward(self, img1, img2):
        return to_ssim(img1, img2)

class mse_loss(nn.Module):
    def __init__(self):
        super(mse_loss, self).__init__()

    def forward(self, img1, img2):
        return to_mseloss(img1, img2)
