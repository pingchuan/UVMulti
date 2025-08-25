import torch
import torch.nn as nn
from torch.nn.functional import feature_alpha_dropout
import torchvision.models as models


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


class DecoderBlock_SP(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1,transpose=False):
        super(DecoderBlock_SP, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = ConvBlock(in_channels // 4, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        if transpose:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(out_channels,
                                   #    in_channels // 4,
                                   out_channels,
                                   #    in_channels // 4,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1,
                                   bias=False),
                # nn.BatchNorm2d(in_channels // 4),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x

class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, kernel_size=3, stride=1, padding=1):
        super(SideoutBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.dropout = nn.Dropout2d(dropout)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x


class encoder(nn.Module):
    def __init__(self, num_classes):
        super(encoder, self).__init__()

        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Encoder
        self.encoder1_conv = resnet.conv1
        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

    def forward(self, x):
        # x 224
        e1 = self.encoder1_conv(x)  # 128
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1_pool = self.maxpool(e1)  # 56
        e2 = self.encoder2(e1_pool)
        e3 = self.encoder3(e2)  # 28
        e4 = self.encoder4(e3)  # 14
        e5 = self.encoder5(e4)  # 7

        return e1, e2, e3, e4, e5


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()


        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)

        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, feature):
        e1, e2, e3, e4, e5 = feature
        d5 = self.decoder5(e5)  # 14
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))  # 28
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))  # 56
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))  # 128
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # 224*224*64
        out1 = self.outconv(d1)  # 224

        return torch.sigmoid(out1)






class ResNet34U1(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super(ResNet34U1, self).__init__()

        self.encoder1 = encoder(num_classes)

        # Decoder
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)

        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, 1),
        )


    def forward(self, x,fp= False):
        e1, e2, e3, e4, e5 = self.encoder1(x)

        d5 = self.decoder5(e5)  # 14
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))  # 28
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))  # 56
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))  # 128
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # 224*224*64
        out1 = self.outconv(d1)  # 224
        if fp:
            return out1, e5#e5最后一层特征
        else:
            return out1

class SegDecoder(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super(SegDecoder, self).__init__()
        # Decoder
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)

        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, 1),
        )


    def forward(self, x,fp= False):
        e1, e2, e3, e4, e5 = x

        d5 = self.decoder5(e5)  # 14
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))  # 28
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))  # 56
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))  # 128
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # 224*224*64
        out1 = self.outconv(d1)  # 224
        if fp:
            return out1, e5#e5最后一层特征
        else:
            return out1


from collections import OrderedDict
class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation