# Name: Bangcheng Wang
# time:
# loss_function.py

import torch
import torch.nn as nn
from pytorch_msssim import MS_SSIM
from torchvision.models import vgg19

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = vgg19(pretrained=True).features
        self.vgg = nn.Sequential(*list(vgg)[:36]).eval()  # 选择前36层
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = self.criterion(x_vgg, y_vgg)
        return loss

class CompositeLoss(nn.Module):
    def __init__(self):
        super(CompositeLoss, self).__init__()
        self.ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=1)
        self.vgg_loss = VGGLoss()

    def forward(self, pred, target):
        ms_ssim_loss = 1 - self.ms_ssim(pred, target)
        vgg_loss = self.vgg_loss(pred, target)
        return ms_ssim_loss + vgg_loss
