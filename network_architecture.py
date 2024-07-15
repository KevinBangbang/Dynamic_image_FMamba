# Name: Bangcheng Wang
# time:
# network_architecture.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_simple import Mamba

# 定义空间特征提取（U-Net）
class SpatialUNet(nn.Module):
    def __init__(self):
        super(SpatialUNet, self).__init__()
        # 使用您提供的 u2net.py 中的定义
        # 请确保 u2net.py 在相同目录或正确导入路径
        from u2net import U2NET
        self.unet = U2NET(in_ch=1, out_ch=1)

    def forward(self, x):
        return self.unet(x)

# 定义时间特征提取（LSTM）
class TemporalFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalFeatureExtractor, self).__init__()
        self.rnn = nn.LSTM(in_channels, out_channels, batch_first=True)

    def forward(self, x):
        x, _ = self.rnn(x)
        return x

# 定义特征融合（FusionMamba Block）
class FusionMambaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionMambaBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm1 = nn.LayerNorm([out_channels, 1, 1])
        self.mamba = Mamba(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.norm2 = nn.LayerNorm([out_channels, 1, 1])

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.norm1(x1)
        x2 = self.conv1(x2)
        x2 = self.norm1(x2)

        y1 = self.mamba(x1)
        y2 = self.mamba(x2)

        y = y1 + y2
        y = self.conv2(y)
        y = self.norm2(y)
        return y

# 定义 FusionMamba 主模型
class FusionMamba(nn.Module):
    def __init__(self):
        super(FusionMamba, self).__init__()
        self.spatial_unet = SpatialUNet()
        self.temporal_extractor = TemporalFeatureExtractor(in_channels=64, out_channels=64)
        self.fusion_mamba_block = FusionMambaBlock(in_channels=64, out_channels=64)

    def forward(self, spatial_input, temporal_input):
        spatial_features = self.spatial_unet(spatial_input)
        temporal_features = self.temporal_extractor(temporal_input)
        fused_output = self.fusion_mamba_block(spatial_features, temporal_features)
        return fused_output
