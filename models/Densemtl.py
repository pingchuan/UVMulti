import math

import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck

import torch.nn as nn
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
class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()


        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)

        self.outconv = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1),
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


class Decoder_s(nn.Module):
    def __init__(self, num_classes):
        super(Decoder_s, self).__init__()


        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        #self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)


    def forward(self, feature):
        _, e2, e3, e4, e5 = feature
        d5 = self.decoder5(e5)  # 14
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))  # 28
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))  # 56
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))  # 128
        #d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # 224*224*64


        return d2

class Decoder_d(nn.Module):
    def __init__(self, num_classes):
        super(Decoder_d, self).__init__()


        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        #self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)



    def forward(self, feature):
        _, e2, e3, e4, e5 = feature
        d5 = self.decoder5(e5)  # 14
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))  # 28
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))  # 56
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))  # 128
        #d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # 224*224*64
          # 224

        return d2
class Decoder_e(nn.Module):
    def __init__(self, num_classes):
        super(Decoder_e, self).__init__()


        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        #self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)



    def forward(self, feature):
        _, e2, e3, e4, e5 = feature
        d5 = self.decoder5(e5)  # 14
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))  # 28
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))  # 56
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))  # 128
        #d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # 224*224*64


        return d2

class SEBlock(nn.Module):
    """ Squeeze-and-excitation block """
    def __init__(self, channels, r=16):
        super(SEBlock, self).__init__()
        self.r = r
        self.squeeze = nn.Sequential(nn.Linear(channels, channels//self.r),
                                     nn.ReLU(),
                                     nn.Linear(channels//self.r, channels),
                                     nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.size()
        squeeze = self.squeeze(torch.mean(x, dim=(2,3))).view(B,C,1,1)
        return torch.mul(x, squeeze)


class SABlock(nn.Module):
    """ Spatial self-attention block """
    def __init__(self, in_channels, out_channels):
        super(SABlock, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                        nn.Sigmoid())
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        attention_mask = self.attention(x)
        features = self.conv(x)
        return torch.mul(features, attention_mask)

class MultiTaskDistillationModule(nn.Module):
    """
        Perform Multi-Task Distillation
        We apply an attention mask to features from other tasks and
        add the result as a residual.
    """
    def __init__(self, tasks, auxilary_tasks, channels):
        super(MultiTaskDistillationModule, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.self_attention = {}

        for t in self.tasks:
            other_tasks = [a for a in self.auxilary_tasks if a != t]
            self.self_attention[t] = nn.ModuleDict({a: SABlock(channels, channels) for a in other_tasks})
        self.self_attention = nn.ModuleDict(self.self_attention)


    def forward(self, x):
        adapters = {t: {a: self.self_attention[t][a](x['features_%s' %(a)]) for a in self.auxilary_tasks if a!= t} for t in self.tasks}
        out = {t: x['features_%s' %(t)] + torch.sum(torch.stack([v for v in adapters[t].values()]), dim=0) for t in self.tasks}
        return out


class DenseD1(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_h = Decoder_d(num_classes=6)
        self.seg_h = Decoder_s(num_classes=6)
        self.enh_h = Decoder_e(num_classes=6)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # e: 用于输出3个通道的卷积层
        self.e = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            self.upsample  # 在输出后添加双线性插值
        )

        # d: 用于输出1个通道的卷积层
        self.d = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            self.upsample  # 在输出后添加双线性插值
        )

        # s: 用于输出6个通道的卷积层
        self.s = nn.Sequential(
            nn.Conv2d(64, 6, kernel_size=3, stride=1, padding=1),
            self.upsample  # 在输出后添加双线性插值
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # Pass on shared encoder
        self.outputs = {}

        self.outputs["enhf"] = self.enh_h(x)
        self.outputs["enh0"] =self.e(self.outputs["enhf"])
        self.outputs["segf"] =self.seg_h(x)
        self.outputs["seg0"] =self.s(self.outputs["segf"])
        self.outputs["dispf"] = self.depth_h(x)
        self.outputs[("disp0", 0)]=self.sigmoid(self.d(self.outputs["dispf"]))
        return self.outputs


class DenseD2(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # depth_h: 用于输出1个通道的卷积层
        self.depth_h = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            self.upsample  # 在输出后添加双线性插值
        )

        # seg_h: 用于输出6个通道的卷积层
        self.seg_h = nn.Sequential(
            nn.Conv2d(64, 6, kernel_size=3, stride=1, padding=1),
            self.upsample  # 在输出后添加双线性插值
        )

        # enh_h: 用于输出3个通道的卷积层
        self.enh_h = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            self.upsample  # 在输出后添加双线性插值
        )
        self.s12 = SxTAM1(in_ch=64,s=4)
        self.s21 = SxTAM1(in_ch=64, s=4)
        self.s13 = SxTAM1(in_ch=64, s=4)
        self.s31 = SxTAM1(in_ch=64, s=4)
        self.s23 = SxTAM1(in_ch=64, s=4)
        self.s32 = SxTAM1(in_ch=64, s=4)

        self.p1 = Project(in_ch=128,out_ch=64)
        self.p2 = Project(in_ch=128, out_ch=64)
        self.p3 = Project(in_ch=128, out_ch=64)


    def forward(self, x1,x2,x3):
        self.outputs = {}


        #x1,x2,x3 =x["segf"],x["dispf"],x["enhf"]
        #x1, x2, x3 = x["segf"], x["dispf"], x["enhf"]
        s1 = self.s21(x1,x2)

        s11 = self.s31(x1,x3)
        s2 = self.s12(x2,x1)
        s22 =self.s32(x2,x3)
        s3 =self.s13(x3,x1)
        s33 =self.s23(x3,x2)

        self.outputs["seg"] = self.seg_h(self.p1(torch.cat((s1,s11), dim=1))+x1)
        self.outputs["enh"] = self.enh_h(self.p2(torch.cat((s2,s22), dim=1))+x2)
        self.outputs[("disp", 0)] = self.depth_h(self.p3(torch.cat((s3,s33), dim=1))+x3)
        return self.outputs


class SxTAM(nn.Module):
    """Spatial cross-Task Attention Module"""
    def __init__(self, in_ch, s, use_alpha):
        super().__init__()

        ## Projection layers
        self.conv_b = nn.Sequential(nn.Conv2d(in_ch, in_ch, 1, bias=False),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(),
                                    nn.Flatten(2))

        self.conv_c = nn.Sequential(nn.Conv2d(in_ch, in_ch, 1, bias=False),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(),
                                    nn.Flatten(2))

        self.conv_d = nn.Sequential(nn.Conv2d(in_ch, in_ch, 1, bias=False),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU())

        self.softmax = nn.Softmax(dim=-1)

        ## Channel-wise weights
        self.use_alpha = use_alpha
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.zeros(1, in_ch, 1, 1))

        self.down = nn.MaxPool2d(s, s)

        self.d = nn.Upsample(scale_factor=1/s)
        self.u = nn.Upsample(scale_factor=s)


    def forward(self, x, y):
        # downscale and flatten spatial dimension
        x_ = self.d(x)
        B = self.conv_b(x_).transpose(1,2)
        C = self.conv_c(self.down(y))
        D = self.conv_d(self.down(y))

        # compute correlation matrix
        # (b, hw_x, c) @ (b, c, hw_y) = (b, hw_x, hw_y) -T-> (b, hw_y, hw_x)
        coeff = math.sqrt(B.size(2))
        corr = self.softmax(B @ C / coeff).transpose(1,2)

        # (b, c, hw_y) @ (b, hw_y, hw_x) = (b, c, hw_x) -view-> (b, c, h_x, w_x)
        out = self.u((D.flatten(2) @ corr).view_as(x_))

        if self.use_alpha:
            out *= self.alpha

        return out

class SxTAM1(nn.Module):
    """Spatial cross-Task Attention Module"""
    def __init__(self, in_ch, s, use_alpha=False):
        super().__init__()

        ## Projection layers
        self.conv_b = nn.Sequential(nn.Conv2d(in_ch, in_ch, 1, bias=False),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(),
                                    nn.Flatten(2))

        self.conv_c = nn.Sequential(nn.Conv2d(in_ch, in_ch, 1, bias=False),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(),
                                    nn.Flatten(2))

        self.conv_d = nn.Sequential(nn.Conv2d(in_ch, in_ch, 1, bias=False),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU())

        self.softmax = nn.Softmax(dim=-1)

        ## Channel-wise weights
        self.use_alpha = use_alpha
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.zeros(1, in_ch, 1, 1))

        self.down = nn.MaxPool2d(s, s)

        self.d = nn.Upsample(scale_factor=1/s)
        self.u = nn.Upsample(scale_factor=s)
        self.sa = SABlock(in_channels=64,out_channels=64)
        self.project = Project(in_ch=128,out_ch=64)
    def forward(self, x, y):
        # downscale and flatten spatial dimension
        x_ = self.d(x)
        B = self.conv_b(x_).transpose(1,2)
        C = self.conv_c(self.down(y))
        D = self.conv_d(self.down(y))
        y1 = self.sa(y)
        # compute correlation matrix
        # (b, hw_x, c) @ (b, c, hw_y) = (b, hw_x, hw_y) -T-> (b, hw_y, hw_x)
        coeff = math.sqrt(B.size(2))
        corr = self.softmax(B @ C / coeff).transpose(1,2)

        # (b, c, hw_y) @ (b, hw_y, hw_x) = (b, c, hw_x) -view-> (b, c, h_x, w_x)
        out = self.u((D.flatten(2) @ corr).view_as(x_))

        if self.use_alpha:
            out *= self.alpha
        out = self.project(torch.cat((out,y1), dim=1))
        return out

class Project(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                         nn.BatchNorm2d(out_ch),
                         nn.ReLU())

class Extractor(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU())