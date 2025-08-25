import torch
import torch.nn as nn
from torch.nn.functional import feature_alpha_dropout
import torchvision.models as models
#from depth_decoder_QTR import Depth_Decoder_QueryTr
from models.depth_decoder_QTR import Depth_Decoder_QueryTr
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

class DecoderBlock_d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super(DecoderBlock_d, self).__init__()

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



class ResNet34U_E(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super(ResNet34U_E, self).__init__()

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
            ConvBlock(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
        )


    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder1(x)

        d5 = self.decoder5(e5)  # 14
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))  # 28
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))  # 56
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))  # 128
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # 224*224*64
        out1 = self.outconv(d1)  # 224
        return out1

class EnhDecoder(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super(EnhDecoder, self).__init__()
        # Decoder
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)

        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            ConvBlock(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        self.outputs = {}
        e1, e2, e3, e4, e5 = x

        d5 = self.decoder5(e5)  # 14
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))  # 28
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))  # 56
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))  # 128
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # 224*224*64
        out1 = self.outconv(d1)
        self.outputs["enh"] = out1
        return self.outputs


class SegDecoder1(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super(SegDecoder1, self).__init__()
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

class PoseCNN(nn.Module):
    def __init__(self, num_input_frames):
        super(PoseCNN, self).__init__()

        self.num_input_frames = num_input_frames

        self.convs = {}
        self.convs[0] = nn.Conv2d(3 * num_input_frames, 16, 7, 2, 3)
        self.convs[1] = nn.Conv2d(16, 32, 5, 2, 2)
        self.convs[2] = nn.Conv2d(32, 64, 3, 2, 1)
        self.convs[3] = nn.Conv2d(64, 128, 3, 2, 1)
        self.convs[4] = nn.Conv2d(128, 256, 3, 2, 1)
        self.convs[5] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[6] = nn.Conv2d(256, 256, 3, 2, 1)

        self.pose_conv = nn.Conv2d(256, 6 * (num_input_frames - 1), 1)

        self.num_convs = len(self.convs)

        self.relu = nn.ReLU(True)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, out):

        for i in range(self.num_convs):
            out = self.convs[i](out)
            out = self.relu(out)

        out = self.pose_conv(out)
        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation


# 定义空间注意力（SPA）
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, scale_factor=0.5):
        super(SpatialAttention, self).__init__()
        self.scale_factor = scale_factor
        # 使用卷积来生成空间注意力权重
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 对输入进行空间注意力的计算
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)  # 合并
        x = self.conv(x)  # 通过卷积
        attention = self.sigmoid(x)

        # 上采样空间注意力图
        return F.interpolate(attention, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

import math
# 定义通道注意力（ECA）
from torch.nn import init
# 定义ECA注意力模块的类

class ECAAttention1(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECAAttention1, self).__init__()
        self.channel = channel
        self.b = b
        self.gamma = gamma

        k_size = self._get_kernel_size()

       # self.compress = nn.Conv2d(in_channels=channel*2, out_channels=channel, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv1d(1,1, kernel_size=k_size, padding=int(k_size // 2),
                              bias=False)

        #self.patchnorm = nn.BatchNorm2d(channel)
        #self.prelu = nn.PReLU(channel)  # PReLU激活函数
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    def _get_kernel_size(self):
        t = int(abs((math.log(self.channel, 2) + self.b) / self.gamma))
        k = t if t % 2 else t + 1
        return k


    def forward(self, x):

        # 对输入进行通道注意力的计算
        #x = self.compress(x)

        #x = self.patchnorm(x)
        #x = self.prelu(x)
        x = self.avg_pool(x)

        x = self.conv(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(x)

        return out

class ECAAttention(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECAAttention, self).__init__()
        self.channel = channel
        self.b = b
        self.gamma = gamma

        k_size = self._get_kernel_size()

        self.compress = nn.Conv2d(in_channels=channel*2, out_channels=channel, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv1d(1,1, kernel_size=k_size, padding=int(k_size // 2),
                              bias=False)

        #self.patchnorm = nn.BatchNorm2d(channel)
        #self.prelu = nn.PReLU(channel)  # PReLU激活函数
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    def _get_kernel_size(self):
        t = int(abs((math.log(self.channel, 2) + self.b) / self.gamma))
        k = t if t % 2 else t + 1
        return k


    def forward(self, x):

        # 对输入进行通道注意力的计算
        x = self.compress(x)

        #x = self.patchnorm(x)
        #x = self.prelu(x)
        x = self.avg_pool(x)

        x = self.conv(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(x)

        return out


class FeatureEnhancement(nn.Module):
    def __init__(self, in_channels_F1, in_channels_F2, in_channels_F3):
        super(FeatureEnhancement, self).__init__()

        # 通道注意力


        # 1x1卷积用于F1'
        self.conv_F1 = nn.Conv2d(in_channels_F1, in_channels_F1, kernel_size=1)
        self.channel = in_channels_F2
        # 两次3x3卷积用于F3'
        #self.conv_F3_1 = nn.Conv2d(in_channels_F3, in_channels_F3, kernel_size=3, padding=1)
        #self.conv_F3_2 = nn.Conv2d(in_channels_F3, in_channels_F3//2, kernel_size=3, padding=1)
        #self.deconv = nn.ConvTranspose2d(
         #   in_channels_F3, in_channels_F3//2, kernel_size=3, stride=1, padding=1)
        # 通道注意力用于融合后的特征
        self.final_ca = ECAAttention1(in_channels_F2)
        self.conv = nn.Conv2d(self.channel,self.channel//2, kernel_size=3, padding=1)
        # 最终降维卷积
        self.final_conv = nn.Conv2d(in_channels_F1 + in_channels_F3, in_channels_F2, kernel_size=3, padding=1)
        self.prelu = nn.PReLU(in_channels_F2)  # PReLU激活函数
        self.patchnorm = nn.BatchNorm2d(in_channels_F2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, F1, F2, F3):
        # Step 1: Extract high-frequency features from F1
        F1_avg = F.adaptive_avg_pool2d(F1, (1, 1))  # Global average pooling
        F1_up = F.interpolate(F1_avg, size=F1.shape[2:], mode='bilinear', align_corners=False)  # 上采样
        F1_high_freq = F1 - F1_up  # F1的高频特征



        # Step 3: Process F1' and F3' for feature fusion
        F1_processed = self.conv_F1(F1_high_freq)  # 对F1'进行1x1卷积
        F1_processed = F.interpolate(F1_processed, scale_factor=0.5, mode='bilinear', align_corners=False)  # 下采样

        #F3_processed = self.conv_F3_1(F3)  # 对F3'进行第一次3x3卷积
        #F3_processed = self.conv_F3_2(F3_processed)  # 对F3'进行第二次3x3卷积
        F3_processed = F.interpolate(F3, scale_factor=2, mode='bilinear', align_corners=False)  # 上采样
        #F3_processed = self.deconv(F3)
        # Step 4: Concatenate F1' and F3' after processing
        fused_features = torch.cat([F1_processed, F3_processed], dim=1)

        fused_features = self.final_conv(fused_features)
        fused_features = self.patchnorm(fused_features)
        fused_features = self.prelu(fused_features)

        # Step 6: Split the channels into two halves and process
        half_C = self.channel // 2
        f1_first_half = fused_features[:, :half_C, :, :]
        f2_first_half = F2[:, :half_C, :, :]

        # 前一半通道相加
        fused_first_half = f1_first_half + f2_first_half

        # 后一半通道拼接
        f1_second_half = fused_features[:, half_C:, :, :]
        f2_second_half = F2[:, half_C:, :, :]
        concatenated_second_half = torch.cat([f1_second_half, f2_second_half], dim=1)

        # 卷积操作
        concatenated_second_half = self.conv(concatenated_second_half)

        # Step 7: 合并处理后的特征
        fused_features = F2 + torch.cat([fused_first_half, concatenated_second_half], dim=1)
        # Step 6: Final convolution to match the output channels of F2
        out = self.final_ca(fused_features)

        # Step 7: Add the enhanced features with F2
        out = out + F2

        return out
# 主模型
class Enhance_A(nn.Module):
    def __init__(self, channel):
        super(Enhance_A, self).__init__()
        self.spa = SpatialAttention(kernel_size=7)  # 空间注意力
        self.eca = ECAAttention(channel=channel)  # 假设F2的通道数为256
        #self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # 示例卷积层

    def forward(self, F1, F2, F3):
        # 对F1进行空间注意力
        F1_space_attention = self.spa(F1)

        # 对F3进行通道注意力
        F3_channel_attention = self.eca(F3)

        # 计算F2' = F2 * 空间注意力 + F2 * 通道注意力 + F2
        F2_space_enhanced = F2 * F1_space_attention  # F2经过空间注意力加权
        F2_channel_enhanced = F2 * F3_channel_attention  # F2经过通道注意力加权

        F2_prime = F2 + F2_space_enhanced + F2_channel_enhanced  # 最终结果

        return F2_prime



class ResNet34UN(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super(ResNet34UN, self).__init__()

        self.encoder1 = encoder(num_classes)
        #self.en1 = Enhance_A(channel=64)
        #self.en2 = Enhance_A(channel=128)
        #self.en3 = Enhance_A(channel=256)
        self.en1 = FeatureEnhancement(64, 64, 128)
        self.en2 = FeatureEnhancement(64, 128, 256)
        self.en3 = FeatureEnhancement(128, 256, 512)
        # Decoder
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)
        self.side_output = nn.Conv2d(512,
                                     num_classes,
                                     kernel_size=1)
        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, 1),
        )


    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder1(x)

        e2_h = self.en1(e1, e2, e3)
        e3_h = self.en2(e2, e3, e4)
        e4_h = self.en3(e3, e4, e5)
        d5 = self.decoder5(e5)  # 14
        d4 = self.decoder4(torch.cat((d5, e4_h), dim=1))  # 28
        d3 = self.decoder3(torch.cat((d4, e3_h), dim=1))  # 56
        d2 = self.decoder2(torch.cat((d3, e2_h), dim=1))  # 128
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # 224*224*64
        out1 = self.outconv(d1)  # 224
        if self.training:
            return out1, self.side_output(e5)#e5最后一层特征
        else:
            return out1
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
class FE1(nn.Module):
    def __init__(self, in_channels_F1, in_channels_F2, in_channels_F3):
        super(FE1, self).__init__()

        # 通道注意力


        # 1x1卷积用于F1'
        self.conv_F1 = nn.Conv2d(in_channels_F1, in_channels_F1, kernel_size=1)
        self.channel = in_channels_F2
        # 两次3x3卷积用于F3'
        #self.conv_F3_1 = nn.Conv2d(in_channels_F3, in_channels_F3, kernel_size=3, padding=1)
        #self.conv_F3_2 = nn.Conv2d(in_channels_F3, in_channels_F3//2, kernel_size=3, padding=1)
        #self.deconv = nn.ConvTranspose2d(
         #   in_channels_F3, in_channels_F3//2, kernel_size=3, stride=1, padding=1)
        # 通道注意力用于融合后的特征
        self.final_ca = ECAAttention1(in_channels_F2)
        self.conv = nn.Conv2d(self.channel,self.channel//2, kernel_size=3, padding=1)
        # 最终降维卷积
        self.final_conv = nn.Conv2d(in_channels_F1 + in_channels_F3, in_channels_F2, kernel_size=3, padding=1)
        self.prelu = nn.PReLU(in_channels_F2)  # PReLU激活函数
        self.patchnorm = nn.BatchNorm2d(in_channels_F2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dw_conv1 = nn.Conv2d(in_channels_F3, in_channels_F3, kernel_size=1, padding=0, groups=in_channels_F3)
        self.bn1 = nn.BatchNorm2d(in_channels_F3)  # Batch Normalization after dw_conv1

        self.dw_conv2 = nn.Conv2d(in_channels_F3, in_channels_F3, kernel_size=7, padding=3, groups=in_channels_F3)
        self.bn2 = nn.BatchNorm2d(in_channels_F3)  # Batch Normalization after dw_conv2

    def forward(self, F1, F2, F3):
        # Step 1: Extract high-frequency features from F1
        F1_avg = F.adaptive_avg_pool2d(F1, (1, 1))  # Global average pooling
        F1_up = F.interpolate(F1_avg, size=F1.shape[2:], mode='bilinear', align_corners=False)  # 上采样
        F1_high_freq = F1 - F1_up  # F1的高频特征



        # Step 3: Process F1' and F3' for feature fusion
        F1_processed = self.conv_F1(F1_high_freq)  # 对F1'进行1x1卷积
        F1_processed = F.interpolate(F1_processed, scale_factor=0.5, mode='bilinear', align_corners=False)  # 下采样

        #F3_processed = self.conv_F3_1(F3)  # 对F3'进行第一次3x3卷积
        #F3_processed = self.conv_F3_2(F3_processed)  # 对F3'进行第二次3x3卷积
        F3_p1 = self.dw_conv1(F3)
        F3_p1 = self.bn1(F3_p1)
        F3_p2 = self.dw_conv2(F3)
        F3_p2 = self.bn2(F3_p2)
        F3_processed = F.interpolate(F3_p2+F3_p1, scale_factor=2, mode='bilinear', align_corners=False)  # 上采样
        #F3_processed = self.deconv(F3)
        # Step 4: Concatenate F1' and F3' after processing
        fused_features = torch.cat([F1_processed, F3_processed], dim=1)

        fused_features = self.final_conv(fused_features)
        fused_features = self.patchnorm(fused_features)
        fused_features = self.prelu(fused_features)

        # Step 6: Split the channels into two halves and process
        half_C = self.channel // 2
        f1_first_half = fused_features[:, :half_C, :, :]
        f2_first_half = F2[:, :half_C, :, :]

        # 前一半通道相加
        fused_first_half = f1_first_half + f2_first_half

        # 后一半通道拼接
        f1_second_half = fused_features[:, half_C:, :, :]
        f2_second_half = F2[:, half_C:, :, :]
        concatenated_second_half = torch.cat([f1_second_half, f2_second_half], dim=1)

        # 卷积操作
        concatenated_second_half = self.conv(concatenated_second_half)

        # Step 7: 合并处理后的特征
        fused_features = F2 + torch.cat([fused_first_half, concatenated_second_half], dim=1)
        # Step 6: Final convolution to match the output channels of F2
        out = self.final_ca(fused_features)

        # Step 7: Add the enhanced features with F2
        out = out + F2

        return out
# 主模型
class ResNet34U2(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super(ResNet34U2, self).__init__()

        self.encoder1 = encoder(num_classes)

        # Decoder
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.fe1 = FE1(in_channels_F1=64, in_channels_F2=64, in_channels_F3=128)
        self.fe2 = FE1(in_channels_F1=64, in_channels_F2=128, in_channels_F3=256)
        self.fe3 = FE1(in_channels_F1=128, in_channels_F2=256, in_channels_F3=512)
        #self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.outconv = nn.Sequential(

            nn.Conv2d(64, num_classes, 1),
        )


    def forward(self, x,fp= False):
        e1, e2, e3, e4, e5 = self.encoder1(x)
        e2_e = self.fe1(e1, e2, e3)
        e3_e = self.fe2(e2, e3, e4)
        e4_e = self.fe3(e3, e4, e5)
        d5 = self.decoder5(e5)  # 14
        d4 = self.decoder4(torch.cat((d5, e4_e), dim=1))  # 28
        d3 = self.decoder3(torch.cat((d4, e3_e), dim=1))  # 56
        d2 = self.decoder2(torch.cat((d3, e2_e), dim=1))  # 128
        #d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # 224*224*64
        out1 = self.upsample(self.outconv(d2))  # 224
        if fp:
            return out1, e5#e5最后一层特征
        else:
            return out1

import torch.nn.functional as F

class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class UpSampleBN_0(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN_0, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    def forward(self, x):

        x = self._net(x)
        x = self.upsample(x)
        return x


class DepthDecoder_bins(nn.Module):
    def __init__(self, num_classes, patch_size,dim_out,embedding_dim,num_heads,query_nums,min_value, max_value, dropout=0.1):
        super(DepthDecoder_bins, self).__init__()
        # Decoder
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)
        self.depth = Depth_Decoder_QueryTr(in_channels=num_classes,patch_size=patch_size,dim_out=dim_out,embedding_dim=
                                           embedding_dim,num_heads=num_heads, query_nums=query_nums,min_val=min_value,max_val=max_value)
        self.outconv = nn.Sequential(
            ConvBlock(64, num_classes, kernel_size=3, stride=1, padding=1),
            #nn.Dropout2d(dropout),
            #ConvBlock(32, 1, kernel_size=3, stride=1, padding=1),
            #nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )


    def forward(self, x):
        self.outputs = {}
        e1, e2, e3, e4, e5 = x

        d5 = self.decoder5(e5)  # 14
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))  # 28
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))  # 56
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))  # 128
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # 224*224*64
        out1 = self.outconv(d1)

        self.outputs[("disp", 0)] = self.depth(out1)

        return self.outputs

class DepthDecoder(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super(DepthDecoder, self).__init__()
        # Decoder
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)

        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            #ConvBlock(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        self.outputs = {}
        e1, e2, e3, e4, e5 = x

        d5 = self.decoder5(e5)  # 14
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))  # 28
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))  # 56
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))  # 128
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # 224*224*64
        out1 = self.outconv(d1)

        self.outputs[("disp", 0)]=self.sigmoid(out1)

        return self.outputs



 #d1 = d1 + torch.clamp(wt, min=0.0, max=1.0)*enh_f


class DepthDecoder_bins_f(nn.Module):
    def __init__(self, num_classes, patch_size,dim_out,embedding_dim,num_heads,query_nums,min_value, max_value, dropout=0.1):
        super(DepthDecoder_bins_f, self).__init__()
        # Decoder
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)
        self.depth = Depth_Decoder_QueryTr(in_channels=num_classes,patch_size=patch_size,dim_out=dim_out,embedding_dim=
                                           embedding_dim,num_heads=num_heads, query_nums=query_nums,min_val=min_value,max_val=max_value)
        self.outconv = nn.Sequential(
            ConvBlock(64, num_classes, kernel_size=3, stride=1, padding=1),
            #nn.Dropout2d(dropout),
            #ConvBlock(32, 1, kernel_size=3, stride=1, padding=1),
            #nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
        self.channel_w = ChannelWeights(dim=64)

    def forward(self, x,enh_f):
        self.outputs = {}
        e1, e2, e3, e4, e5 = x

        d5 = self.decoder5(e5)  # 14
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))  # 28
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))  # 56
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))  # 128
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # 224*224*64
        #d1 = self.channel_w(d1, enh_f)
        d1 = self.channel_w(d1, enh_f)
        out1 = self.outconv(d1)

        self.outputs[("disp", 0)] = self.depth(out1)

        return self.outputs



class EnhDecoder_f(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super(EnhDecoder_f, self).__init__()
        # Decoder
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)

        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            ConvBlock(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        self.outputs = {}
        e1, e2, e3, e4, e5 = x

        d5 = self.decoder5(e5)  # 14
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))  # 28
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))  # 56
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))  # 128
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # 224*224*64
        out1 = self.outconv(d1)
        self.outputs["enh"] = out1
        self.outputs["enh_f"] = d1
        return self.outputs

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


    def forward(self, x):
        self.outputs = {}
        e1, e2, e3, e4, e5 = x

        d5 = self.decoder5(e5)  # 14
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))  # 28
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))  # 56
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))  # 128
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # 224*224*64
        out1 = self.outconv(d1)  # 224
        self.outputs["seg"] = out1
        return self.outputs


# Feature Rectify Module
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim * 4, self.dim * 4 // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim * 4 // reduction, self.dim * 2),
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        #x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x1).view(B, self.dim)
        max = self.max_pool(x1).view(B, self.dim)
        y = torch.cat((avg, max), dim=1) # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B,  self.dim, 1, 1).permute(0, 1, 2, 3)  # B C 1 1
        enh_f = x2*channel_weights
        wt = cosine_similarity(x1, enh_f)
        d1 = x1 + torch.clamp(wt, min=0.0, max=1.0)*enh_f
        return d1



def cosine_similarity(x_k, x_t):
    """
    计算两个特征图 x_k 和 x_t 在每个位置上的余弦相似度。
    输入:
        x_k: 特征图1，形状为 [B, C, H, W]
        x_t: 特征图2，形状为 [B, C, H, W]
    输出:
        wt: 每个像素的余弦相似度，形状为 [B, 1, H, W]
    """
    # 计算每个位置的特征向量的 L2 范数
    norm_k = torch.norm(x_k, p=2, dim=1, keepdim=True)  # [B, 1, H, W]
    norm_t = torch.norm(x_t, p=2, dim=1, keepdim=True)  # [B, 1, H, W]

    # 计算点积 (B, C, H, W)
    dot_product = torch.sum(x_k * x_t, dim=1, keepdim=True)  # [B, 1, H, W]

    # 计算余弦相似度
    wt = dot_product / (norm_k * norm_t + 1e-8)  # 防止除零错误

    return wt


class DwConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DwConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x



# Transformer Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)

        self.scale = in_channels ** 0.5

    def forward(self, query, key, value):
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, V)

        return output


# 主模型类
class FEDS(nn.Module):
    def __init__(self, in_channels, out_channels_F2):
        super(FEDS, self).__init__()

        # F1 和 F2 的交叉注意力
        self.fc_F1 = nn.Linear(in_channels, in_channels)
        self.fc_F2_k = nn.Linear(in_channels, in_channels)
        self.fc_F2_v = nn.Linear(in_channels, in_channels)
        # F1' 和 F2 计算注意力
        self.attention_block_F1_F2 = AttentionBlock(in_channels)

        # F3 的处理
        self.fc_F3 = nn.Linear(in_channels, in_channels)
        self.dwconv_F3 = DwConv(in_channels, in_channels)

        # 最后的融合
        self.fc_F2_F3 = nn.Linear(in_channels, in_channels)
        self.conv_out = nn.Conv2d(in_channels * 2, out_channels_F2, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels_F2)
        self.prelu = nn.PReLU()

    def forward(self, F1, F2, F3):
        # Step 1: F1 和 F2 计算余弦相似度得到W，然后更新 F1
        sim = cosine_similarity(F1, F2)  # 计算余弦相似度

        F1_prime = F1 + torch.clamp(sim, min=0.0, max=1.0)*F1 # F1增强

        # Step 2: F1' 和 F2 经过不同的线性层，得到 Q 和 K, V
        F1_prime_linear = self.fc_F1(F1_prime)
        F2_k = self.fc_F2_k(F2)  # 使用相同的线性层处理 F2
        F2_v = self.fc_F2_v(F2)
        # 使用 Transformer 注意力机制
        Fa = self.attention_block_F1_F2(F1_prime_linear, F2_k, F2_v)

        # Step 3: F3 经过线性层和 DwConv 得到 F3'
        F3_linear = self.fc_F3(F3)
        F3_prime = self.dwconv_F3(F3_linear)

        # Step 4: F2 与 F3' 点乘得到 Fb
        F2_F3 = self.fc_F2_F3(F2)
        Fb = F2_F3 * F3_prime

        # Step 5: 拼接 Fa 和 Fb
        fused_features = torch.cat([Fa, Fb], dim=1)

        # Step 6: 最后的卷积操作
        out = self.conv_out(fused_features)
        out = self.bn(out)
        out = self.prelu(out)

        return out
class SegDecoder_f(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super(SegDecoder_f, self).__init__()
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
        self.channel_w = ChannelWeights(dim=64)

    def forward(self, x,enh_f):
        self.outputs = {}
        e1, e2, e3, e4, e5 = x

        d5 = self.decoder5(e5)  # 14
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))  # 28
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))  # 56
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))  # 128
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # 224*224*64
        #wt = cosine_similarity(d1, enh_f)
        #d1 = d1 + torch.clamp(wt, min=0.0, max=1.0)*enh_f
        d1 = self.channel_w(d1,enh_f)
        out1 = self.outconv(d1)  # 224
        self.outputs["seg"] = out1
        return self.outputs

import numpy as np
import cv2
def optimize_segmentation_with_pose(mask_current, mask_prev, mask_next, pose_prev_to_current, pose_current_to_next,
                                    alpha=0.33, beta=0.33, gamma=0.33):
    """
    通过前后帧位姿估计优化当前帧的分割掩码，输入掩码为多类别掩码（每个像素为类别标签），
    通过位姿变换将前后帧掩码变换到当前帧视角，并进行加权融合。

    参数:
        mask_current: 当前帧的分割掩码 (H, W)，每个像素值为类别标签（0, 1, 2...）
        mask_prev: 前一帧的分割掩码 (H, W)
        mask_next: 后一帧的分割掩码 (H, W)
        pose_prev_to_current: 前一帧到当前帧的位姿变换矩阵 (4, 4)
        pose_current_to_next: 当前帧到后一帧的位姿变换矩阵 (4, 4)
        alpha, beta, gamma: 掩码融合时的权重，默认为 1/3 权重

    返回:
        optimized_mask: 优化后的当前帧分割掩码 (H, W)
    """

    def transform_mask(mask, pose_matrix):
        """
        通过位姿矩阵将掩码从一个视角变换到另一个视角
        """
        H, W = mask.shape
        y, x = np.indices((H, W))
        homogenous_coords = np.vstack((x.flatten(), y.flatten(), np.ones((1, x.size))))
        transformed_coords = np.dot(pose_matrix, homogenous_coords)  # 齐次坐标变换
        transformed_coords /= transformed_coords[2, :]  # 归一化

        transformed_x = transformed_coords[0, :].reshape(H, W)
        transformed_y = transformed_coords[1, :].reshape(H, W)

        # 使用 OpenCV 的 remap 进行双线性插值
        transformed_mask = cv2.remap(mask, transformed_x.astype(np.float32), transformed_y.astype(np.float32),
                                     interpolation=cv2.INTER_NEAREST)
        return transformed_mask

    def fuse_masks(mask_current, mask_prev, mask_next, alpha=0.33, beta=0.33, gamma=0.33):
        """
        融合多个掩码（加权平均）
        """
        # 加权融合，保留掩码类别
        fused_mask = np.zeros_like(mask_current)
        for y in range(mask_current.shape[0]):
            for x in range(mask_current.shape[1]):
                # 获取每个位置的类别
                category_values = [mask_current[y, x], mask_prev[y, x], mask_next[y, x]]
                # 找到出现频率最多的类别
                unique, counts = np.unique(category_values, return_counts=True)
                most_frequent_category = unique[np.argmax(counts)]
                fused_mask[y, x] = most_frequent_category
        return fused_mask

    # 变换前后帧的分割掩码到当前帧的视角
    mask_prev_transformed = transform_mask(mask_prev, pose_prev_to_current)
    mask_next_transformed = transform_mask(mask_next, pose_current_to_next)

    # 融合当前帧、前一帧和后一帧的分割掩码
    optimized_mask = fuse_masks(mask_current, mask_prev_transformed, mask_next_transformed, alpha, beta, gamma)

    return optimized_mask

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