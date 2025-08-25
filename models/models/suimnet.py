import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MyUpSample2X1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(MyUpSample2X1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = F.relu(self.batch_norm(self.conv(x)))
        x = torch.cat([x, skip], dim=1)  # Concatenate along channel dimension
        return x
class MyUpSample2X(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(MyUpSample2X, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, skip):
        # 先对 x 进行上采样
        x = self.upsample(x)
        x = F.relu(self.batch_norm(self.conv(x)))

        # 进行裁剪，使得 x 和 skip 的尺寸一致
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = F.interpolate(x, size=(skip.size(2), skip.size(3)), mode='bilinear', align_corners=True)

        # 将两个张量沿通道维度拼接
        x = torch.cat([x, skip], dim=1)  # Concatenate along channel dimension
        return x


class SUIM_Net_VGG16_Encoder(nn.Module):
    def __init__(self, n_classes):
        super(SUIM_Net_VGG16_Encoder, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features

        # 获取VGG16的池化层
        self.pool1 = self.features[:6]  # block1
        self.pool2 = self.features[6:13]  # block2
        self.pool3 = self.features[13:23]  # block3
        self.pool4 = self.features[23:33]  # block4

        # 解码器
        self.decoder1 = MyUpSample2X(512, 512)
        self.decoder2 = MyUpSample2X(512 + 512, 256)
        self.decoder3 = MyUpSample2X(256 + 256, 128)
        self.decoder4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.final_conv = nn.Conv2d(256, n_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # 编码器部分
        x1 = self.pool1(x)
        x2 = self.pool2(x1)
        x3 = self.pool3(x2)
        x4 = self.pool4(x3)

        # 解码器部分
        dec1 = self.decoder1(x4, x3)
        dec2 = self.decoder2(dec1, x2)
        dec3 = self.decoder3(dec2, x1)
        dec4 = self.decoder4(dec3)

        out = self.final_conv(dec4)
        return out
class SUIM_Net(nn.Module):
    def __init__(self, num_classes):
        super(SUIM_Net, self).__init__()
        self.encoder = SUIM_Net_VGG16_Encoder(num_classes)

    def forward(self, x):
        out = self.encoder(x)
        return out


class RSB(nn.Module):
    def __init__(self, in_channels, kernel_size, filters, strides=1, skip=True):
        super(RSB, self).__init__()
        f1, f2, f3, f4 = filters

        # Sub-block 1
        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=(1, 1), stride=strides)
        self.bn1 = nn.BatchNorm2d(f1)

        # Sub-block 2
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(f2)

        # Sub-block 3
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=(1, 1))
        self.bn3 = nn.BatchNorm2d(f3)

        if skip:
            self.skip = None
        else:
            self.skip = nn.Conv2d(in_channels, f4, kernel_size=(1, 1), stride=strides)
            self.bn_skip = nn.BatchNorm2d(f4)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Sub-block 1
        x1 = self.relu(self.bn1(self.conv1(x)))

        # Sub-block 2
        x2 = self.relu(self.bn2(self.conv2(x1)))

        # Sub-block 3
        x3 = self.bn3(self.conv3(x2))

        # Skip connection
        if self.skip is None:
            shortcut = x
        else:
            shortcut = self.bn_skip(self.skip(x))

        # Add skip connection and activation
        out = self.relu(x3 + shortcut)
        return out


class SUIM_Encoder_RSB(nn.Module):
    def __init__(self, in_channels=3, out_channels=5, image_size=(320, 240)):
        super(SUIM_Encoder_RSB, self).__init__()
        self.image_size = image_size
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=1),
            nn.ReLU(inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2)
        )
        self.rsb1 = RSB(64, 3, [64, 64, 128, 128], strides=2, skip=False)
        self.rsb2 = RSB(128, 3, [64, 64, 128, 128], skip=True)
        self.rsb3 = RSB(128, 3, [64, 64, 128, 128], skip=True)

        self.rsb4 = RSB(128, 3, [128, 128, 256, 256], strides=2, skip=False)
        self.rsb5 = RSB(256, 3, [128, 128, 256, 256], skip=True)
        self.rsb6 = RSB(256, 3, [128, 128, 256, 256], skip=True)
        self.rsb7 = RSB(256, 3, [128, 128, 256, 256], skip=True)

    def forward(self, x):
        # Encoder block 1
        enc_1 = self.encoder1(x)

        # Encoder block 2
        x = self.encoder2(enc_1)
        enc_2 = self.rsb1(x)
        enc_2 = self.rsb2(enc_2)
        enc_2 = self.rsb3(enc_2)

        # Encoder block 3
        enc_3 = self.rsb4(enc_2)
        enc_3 = self.rsb5(enc_3)
        enc_3 = self.rsb6(enc_3)
        enc_3 = self.rsb7(enc_3)

        return enc_1, enc_2, enc_3


class SUIM_Decoder_RSB(nn.Module):
    def __init__(self, enc_inputs, n_classes):
        super(SUIM_Decoder_RSB, self).__init__()

        enc_1, enc_2, enc_3 = enc_inputs

        self.upconv1 = self.upsample_block(enc_3, enc_2, 256)
        self.upconv2 = self.upsample_block(self.upconv1, enc_1, 128)
        self.upconv3 = self.upsample_block(self.upconv2, enc_1, 64)

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def upsample_block(self, layer_input, skip_input, filters):
        upsampled = F.interpolate(layer_input, scale_factor=2, mode='bilinear', align_corners=True)
        upsampled = nn.Conv2d(upsampled.size(1), filters, kernel_size=3, padding=1)(upsampled)
        upsampled = F.relu(upsampled)
        upsampled = torch.cat([upsampled, skip_input], dim=1)
        return upsampled

    def forward(self, enc_inputs):
        enc_1, enc_2, enc_3 = enc_inputs

        dec_1 = self.upconv1(enc_3, enc_2, 256)
        dec_2 = self.upconv2(dec_1, enc_1, 128)
        dec_3 = self.upconv3(dec_2, enc_1, 64)

        out = self.final_conv(dec_3)
        out = self.sigmoid(out)
        return out





