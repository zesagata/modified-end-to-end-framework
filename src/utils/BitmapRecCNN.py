import torch
import torch.nn as nn


class BitmapRecCNN(nn.Module):
    def __init__(self, channel,interpolate_size=2,mode='bicubic,deep=5):
        super(BitmapRecCNN, self).__init__()
        self.deconv1 = nn.Conv2d(channel, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64, affine=False)
        self.deconv_n = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn_n = nn.BatchNorm2d(64, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.interpolate_size = interpolate_size
        self.mode = mode
        self.deep = 5

    def forward(self, x):
        x = self.interpolate(x)
        out = self.relu(self.deconv1(x))
        for _ in range(self.deep):
            out = self.relu(self.bn_n(self.deconv_n(out)))
        out = self.relu(self.deconv3(out))
        final = out.add(x)
        rounding = self.rounding_layer(final)
        return final,out,rounding

    def interpolate(self,x):
        return nn.functional.interpolate(input=x, scale_factor=self.interpolate_size, mode=self.mode,
                                         align_corners=False)

    def rounding_layer(self,batch_image):
        return batch_image