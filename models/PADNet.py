import torch
import torch.nn as nn
from torch.nn.functional import feature_alpha_dropout
import torchvision.models as models
import torch.nn.functional as F
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super(DecoderBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = ConvBlock(in_channels // 4, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x


class Enh_h(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super(Enh_h, self).__init__()
        # Decoder


        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            ConvBlock(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        self.outputs = {}


        enh_f, _, _ = x
        out1 = self.outconv(enh_f)
        self.outputs["enh"] = out1
        return self.outputs

class Depth_h(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super(Depth_h, self).__init__()
        # Decoder

        self.outconv = nn.Sequential(
            ConvBlock(64*3, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            ConvBlock(32, 1, kernel_size=3, stride=1, padding=1),
            #nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        self.outputs = {}

        enh_f, s_f, d_f = x
        d1 = torch.cat((enh_f, s_f, d_f), dim=1)
        out1 = self.outconv(d1)

        self.outputs[("disp", 0)] = self.sigmoid(out1)

        return self.outputs

class Seg_f(nn.Module):
    def __init__(self):
        super(Seg_f, self).__init__()
        # Decoder
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)


    def forward(self, x):
        e1, e2, e3, e4, e5 = x

        d5 = self.decoder5(e5)  # 14
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))  # 28
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))  # 56
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))  # 128
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # 224*224*64

        return d1
class Seg_h(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super(Seg_h, self).__init__()
        # Decoder

        self.outconv = nn.Sequential(
            ConvBlock(64*3, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, 1),
        )
        #self.channel_w = ChannelWeights(dim=64)


    def forward(self, x):
        self.outputs = {}
        enh_f, s_f, d_f = x
        d1 = torch.cat((enh_f, s_f, d_f), dim=1)
        #wt = cosine_similarity(d1, enh_f)
        #d1 = d1 + torch.clamp(wt, min=0.0, max=1.0)*enh_f
        #d1 = self.enh_eds(enh_f, s_f, d_f)
        #d1 = self.channel_w(d1,enh_f)
        out1 = self.outconv(d1)  # 224
        self.outputs["seg"] = out1
        return self.outputs
