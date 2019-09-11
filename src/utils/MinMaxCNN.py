import torch
import torch.nn as nn


class MinMaxCNN(nn.Module):
    def __init__(self, channel,interpolate_size=2,mode='bicubic,deep=3):
        super(MinMaxCNN, self).__init__()
        self.deconv1 = nn.Conv2d(channel, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64, affine=False)
        self.deconv_n = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn_n = nn.BatchNorm2d(64, affine=False)
        self.deconv3 = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.deconv1(x))
        out = self.relu(self.bn_n(self.deconv_n(out)))
        out = self.deconv3(out)
        return out