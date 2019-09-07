import torch.nn as nn
class RecCNN(nn.Module):
    def __init__(self, channel):
        super(RecCNN, self).__init__()
        self.deconv1 = nn.Conv2d(channel, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64, affine=False)
        self.deconv_n = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn_n = nn.BatchNorm2d(64, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.deconv1(x))
        for _ in range(18):
            out = self.relu(self.bn_n(self.deconv_n(out)))
        out = self.deconv3(out)
        final = out.add(x)
        return final, out
