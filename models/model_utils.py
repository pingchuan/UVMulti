# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from modulefinder import Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import re
class ConvBNAct(nn.Sequential):
    def __init__(self, channels_in, channels_out, kernel_size,
                 activation=nn.ReLU(inplace=True), dilation=1, stride=1):
        super(ConvBNAct, self).__init__()
        padding = kernel_size // 2 + dilation - 1
        self.add_module('conv', nn.Conv2d(channels_in, channels_out,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          bias=False,
                                          dilation=dilation,
                                          stride=stride))
        self.add_module('bn', nn.BatchNorm2d(channels_out))
        self.add_module('act', activation)


class ConvBN(nn.Sequential):
    def __init__(self, channels_in, channels_out, kernel_size):
        super(ConvBN, self).__init__()
        self.add_module('conv', nn.Conv2d(channels_in, channels_out,
                                          kernel_size=kernel_size,
                                          padding=kernel_size // 2,
                                          bias=False))
        self.add_module('bn', nn.BatchNorm2d(channels_out))

class Se(nn.Module):
    def __init__(self, channel, reduction=16, activation=nn.ReLU()):
        super(Se, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel // reduction)
        self.activation = activation
        self.conv2 = nn.Conv2d(channel // reduction, 1, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x

class FRS(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(FRS, self).__init__()
        # 注意力机制的卷积层
        self.se1 = Se(channel,reduction=reduction)
        self.se2 = Se(channel,reduction=reduction)
        self.center_bias = nn.Parameter(torch.ones(1, 1, 128, 128))  # 中心偏置参数

    def forward(self, x,x2):
        # 1. 使用卷积层将输入特征图压缩到单通道
        x_r = self.se1(x)
        x_d = self.se2(x2)
        # 2. 对单通道的特征图进行傅里叶变换（频域分析）
        x_fft_r = torch.fft.fft2(x_r)
        x_fft_shift_r = torch.fft.fftshift(x_fft_r)  # 将低频分量移到中心

        # 3. 获取幅度谱
        magnitude_spectrum_r = torch.log(1 + torch.abs(x_fft_shift_r))

        # 4. 求显著性图
        saliency_map_r = magnitude_spectrum_r  # 已经是单通道 [b, 1, h, w]

        # 5. 应用中心偏置
        saliency_map_r = saliency_map_r * self.center_bias

        # 6. 归一化显著性图
        saliency_map_r = (saliency_map_r - saliency_map_r.min()) / (saliency_map_r.max() - saliency_map_r.min())
        x_fft_d = torch.fft.fft2(x_d)
        x_fft_shift_d= torch.fft.fftshift(x_fft_d)  # 将低频分量移到中心

        # 3. 获取幅度谱
        magnitude_spectrum_d = torch.log(1 + torch.abs(x_fft_shift_d))

        # 4. 求显著性图
        saliency_map_d = magnitude_spectrum_d  # 已经是单通道 [b, 1, h, w]

        # 5. 应用中心偏置
        saliency_map_d = saliency_map_d * self.center_bias

        # 6. 归一化显著性图
        saliency_map_d = (saliency_map_d - saliency_map_d.min()) / (saliency_map_d.max() - saliency_map_d.min())
        saliency_map_r = (saliency_map_r >= 0.5).float()
        saliency_map_d = (saliency_map_d >= 0.5).float()
        return saliency_map_r, saliency_map_d



#编码器特征四组融合到第一次解码器的输入APFABlock
class APFABlock(nn.Module):
    def __init__(self, in_channels=960, out_channels=512, atrous_rates=[1, 2, 4, 8]):
        super(APFABlock, self).__init__()
        self.weights = nn.Parameter(torch.full((4,), 0.25)) # weights for each resolution [w1, w2, w3, w4]
        self.branches = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=3,
                       dilation=rate, padding=rate) for rate in atrous_rates]
        )
        self.conv_1x1 = nn.Conv2d(len(atrous_rates) * out_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2, x3, x4):
        # Weight and combine feature maps from different stages
        x1_weighted = self.weights[0] * x1
        x2_weighted = self.weights[1] * x2
        x3_weighted = self.weights[2] * x3
        x4_weighted = self.weights[3] * x4

        # 调整空间尺寸到与 x4 相同的尺寸 (16, 16)
        x1_weighted = F.interpolate(x1_weighted, size=(16, 16), mode='bilinear', align_corners=False)
        x2_weighted = F.interpolate(x2_weighted, size=(16, 16), mode='bilinear', align_corners=False)
        x3_weighted = F.interpolate(x3_weighted, size=(16, 16), mode='bilinear', align_corners=False)
        # Apply dilated convolutions for each branch
        x_combined = torch.cat((x1_weighted, x2_weighted, x3_weighted, x4_weighted), dim=1)

        # Apply dilated convolutions for each branch
        branch_outputs = [branch(x_combined) for branch in self.branches]

        # Concatenate the outputs along the channel dimension
        x_new_combined = torch.cat(branch_outputs, dim=1)

        # Pass the combined map through a 1x1 convolution
        x_out = self.conv_1x1(x_new_combined)

        return x_out


class RGB_ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(RGB_ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim * 2, self.dim * 2 // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim * 2 // reduction, self.dim),
                    nn.Sigmoid())

    def forward(self, x1):
        B, _, H, W = x1.shape

        avg = self.avg_pool(x1).view(B, self.dim)
        max = self.max_pool(x1).view(B, self.dim)
        y = torch.cat((avg, max), dim=1) # B 2C
        y = self.mlp(y).view(B, self.dim, 1)
        channel_weights = y.reshape(B, self.dim, 1, 1)
        return channel_weights

class Depth_SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(Depth_SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
                    nn.Conv2d(self.dim, self.dim // reduction, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dim // reduction, 1, kernel_size=1),
                    nn.Sigmoid())

    def forward(self, x1):
        B, _, H, W = x1.shape

        spatial_weights = self.mlp(x1).reshape(B, 1, H, W)
        return spatial_weights


class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels)
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x):
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out


class ChannelWeights_new(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(ChannelWeights_new, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1) #[B,C,1,1]
        weighting = self.fc(weighting)

        return weighting

class SPALayer_new(nn.Module):
    def __init__(self,channel, reduction=16,activation=nn.ReLU(inplace=True)):
        super(SPALayer_new, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.avg_pool7 = nn.AdaptiveAvgPool2d(7)
        self.weight = Parameter(torch.ones(1, 3, 1, 1, 1))
        self.transform = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.BatchNorm2d(channel // reduction),
            activation,
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool1(x)
        y2 = self.avg_pool4(x)
        y4 = self.avg_pool7(x)
        y = torch.cat(
            [y4.unsqueeze(dim=1),
             F.interpolate(y2, size=[7,7]).unsqueeze(dim=1),
             F.interpolate(y1, size=[7,7]).unsqueeze(dim=1)],
            dim=1
        )
        y = (y * self.weight).sum(dim=1, keepdim=False)
        y = self.transform(y)
        y = F.interpolate(y, size=x.size()[2:])

        return y


class RCAB(nn.Module):
    def __init__(self, channel, reduction=16, activation=nn.ReLU(inplace=True)):
        super(RCAB, self).__init__()

        # 卷积层
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

        # 通道注意力机制
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.activation = activation

    def forward(self, x):
        # 残差连接
        residual = x
        x = self.conv(x)
        # 通道注意力机制
        ca_weight = F.adaptive_avg_pool2d(x, 1)  # [B,C,1,1]
        ca_weight = self.ca(ca_weight)
        x = x * ca_weight  # 逐元素相乘
        # 残差连接，最后经过激活函数
        out = self.activation(x + residual)
        return out
#最新的特征融合模块
class FeatureFusionModule3(nn.Module):
    def __init__(self, dim, reduction=1,  norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.rgb_channel = ChannelWeights_new(channel=dim)
        self.rgb_spatial = SPALayer_new(channel=dim)
        self.depth_channel = ChannelWeights_new(channel=dim)
        self.depth_spatial = SPALayer_new(channel=dim)
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction,
                                        norm_layer=norm_layer)
        self.bn1 = norm_layer(dim)
        self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        batch_size, c, h, w = x1.size()
        y1, u1 = self.relu(self.channel_proj1(x1.flatten(2).transpose(1, 2))).chunk(2, dim=-1)
        y2, u2 = self.relu(self.channel_proj2(x2.flatten(2).transpose(1, 2))).chunk(2, dim=-1)

        w_y1 = self.rgb_channel(y1.transpose(1, 2).view(batch_size, c, h, w))
        w_u1 = self.rgb_spatial(u1.transpose(1, 2).view(batch_size, c, h, w))
        w_y2 = self.depth_channel(y2.transpose(1, 2).view(batch_size, c, h, w))
        w_u2 = self.depth_spatial(u2.transpose(1, 2).view(batch_size, c, h, w))

        x_rgb = x1*w_y1 + x1*w_y2
        x_depth = x2*w_u2 + x2*w_u1

        #merge = torch.cat((x_rgb, x_depth), dim=1).squeeze(0)
        merge = torch.cat((x_rgb, x_depth), dim=1)
        merge = self.channel_emb(merge)

        return merge


class FeatureFusionModule2(nn.Module):
    def __init__(self, dim, reduction=1,  norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.channel_weights = RGB_ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = Depth_SpatialWeights(dim=dim, reduction=reduction)
        self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction,
                                        norm_layer=norm_layer)
        self.bn1 = norm_layer(dim)
        self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        w1 = self.channel_weights(x1)
        w2 = self.spatial_weights(x2)

        x1 = x1 + x1*w2

        #jia bn层 激活函数
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2 = x2 + x2*w1
        x2 = self.bn1(x2)
        x2 = self.relu(x2)

        merge = torch.cat((x1, x2), dim=1).squeeze(0)

        merge = self.channel_emb(merge)

        return merge

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
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1) # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4) # 2 B C 1 1
        return channel_weights


from timm.models.layers import trunc_normal_
import math
class FeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1):
        super(FeatureRectifyModule, self).__init__()

        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):

        #channel_weights = self.channel_weights(x1, x2)
        channel_weights = self.channel_weights(x1, x2)
        out_x1 = x1 + channel_weights[1] * x2
        out_x2 = x2 + channel_weights[0] * x1

        return out_x1, out_x2


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
#ICLR2024 Dformer
class FRM2(nn.Module):
    def __init__(self, dim, norm_cfg=dict(type='SyncBN', requires_grad=True), drop_depth=False):
        super().__init__()



        self.q = nn.Linear(dim, dim)
        self.q_cut = nn.Linear(dim, dim )
        self.a = nn.Linear(dim, dim)
        self.l = nn.Linear(dim, dim)
        self.conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.e_conv = nn.Conv2d(dim // 2, dim // 2, 7, padding=3, groups=dim // 2)
        self.e_fore = nn.Linear(dim , dim // 2)
        self.e_back = nn.Linear(dim // 2, dim )

        self.proj = nn.Linear(dim * 2, dim)
        if not drop_depth:
            self.proj_e = nn.Linear(dim * 2, dim )


        self.act = nn.GELU()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.norm_e = LayerNorm(dim , eps=1e-6, data_format="channels_last")
        self.drop_depth = drop_depth

    def forward(self, x, x_e):
        x = x.permute(0, 2, 3, 1)
        x_e = x_e.permute(0, 2, 3, 1)
        x = self.norm(x)
        x_e = self.norm_e(x_e)
        q = self.q(x)
        cutted_x = self.q_cut(x)
        x = self.l(x).permute(0, 3, 1, 2)
        x = self.act(x)

        a = self.conv(x)
        a = a.permute(0, 2, 3, 1)
        a = self.a(a)
        x_e = self.e_back(self.e_conv(self.e_fore(x_e).permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        cutted_x = cutted_x * x_e
        x = q * a
        x = torch.cat([x, cutted_x], dim=3)
        if not self.drop_depth:
            x_e = self.proj_e(x)
        x = self.proj(x)
        x = x.permute(0, 3, 1, 2)
        x_e = x_e.permute(0, 3, 1, 2)
        return x, x_e

class SESP(nn.Module):
    def __init__(self, channel, reduction=16, activation=nn.ReLU(inplace=True), kernel_size=7):
        super(SESP, self).__init__()
        # Channel Attention (Squeeze and Excitation)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        channel_weighting = F.adaptive_avg_pool2d(x, 1)
        channel_weighting = self.channel_attention(channel_weighting)
        y = x * channel_weighting

        # Spatial Attention
        avg_out = torch.mean(y, dim=1, keepdim=True)
        max_out, _ = torch.max(y, dim=1, keepdim=True)
        spatial_weighting = torch.cat([avg_out, max_out], dim=1)
        spatial_weighting = self.spatial_attention(spatial_weighting)
        out = y * spatial_weighting

        return out


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1) #[B,C,1,1]
        weighting = self.fc(weighting)
        y = x * weighting
        return y


class SqueezeAndExcitationTensorRT(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitationTensorRT, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # TensorRT restricts the maximum kernel size for pooling operations
        # by "MAX_KERNEL_DIMS_PRODUCT" which leads to problems if the input
        # feature maps are of large spatial size
        # -> workaround: use cascaded two-staged pooling
        # see: https://github.com/onnx/onnx-tensorrt/issues/333
        if x.shape[2] > 120 and x.shape[3] > 160:
            weighting = F.adaptive_avg_pool2d(x, 4)
        else:
            weighting = x
        weighting = F.adaptive_avg_pool2d(weighting, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y

#*SpatialGroupEnhance
class SpatialGroupEnhance(nn.Module):
    '''
        Spatial Group-wise Enhance: Improving Semantic Feature Learning in Convolutional Networks
        https://arxiv.org/pdf/1905.09646.pdf
    '''
    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x

#*Triplet Attention
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


# class SpatialGate(nn.Module):
#     def __init__(self):
#         super(SpatialGate, self).__init__()

#         self.channel_pool = ChannelPool()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
#             nn.BatchNorm2d(1)
#         )
#         self.sigmod = nn.Sigmoid()

#     def forward(self, x):
#         out = self.conv(self.channel_pool(x))
#         return out * self.sigmod(out)


class TripletAttention(nn.Module):
    def __init__(self, spatial=True):
        super(TripletAttention, self).__init__()
        self.spatial = spatial
        self.height_gate = SpatialGate()
        self.width_gate = SpatialGate()
        if self.spatial:
            self.spatial_gate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.height_gate(x_perm1)
        x_out1 = x_out1.permute(0, 2, 1, 3).contiguous()

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.width_gate(x_perm2)
        x_out2 = x_out2.permute(0, 3, 2, 1).contiguous()

        if self.spatial:
            x_out3 = self.spatial_gate(x)
            return (1/3) * (x_out1 + x_out2 + x_out3)
        else:
            return (1/2) * (x_out1 + x_out2)



#*CBAM
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, reduction=16,activation=nn.ReLU(inplace=True)):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = channel // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            activation,
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    """
        CBAM: Convolutional Block Attention Module
        https://arxiv.org/pdf/1807.06521.pdf
    """
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

#*BAM
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.gate_activation =nn.ReLU(inplace=True)
        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', Flatten() )
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
        self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d( in_tensor, in_tensor.size(2), stride=in_tensor.size(2) )
        return self.gate_c( avg_pool ).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)

class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module( 'gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module( 'gate_s_bn_reduce0',	nn.BatchNorm2d(gate_channel//reduction_ratio) )
        self.gate_s.add_module( 'gate_s_relu_reduce0',nn.ReLU() )
        for i in range( dilation_conv_num ):
            self.gate_s.add_module( 'gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
						padding=dilation_val, dilation=dilation_val) )
            self.gate_s.add_module( 'gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio) )
            self.gate_s.add_module( 'gate_s_relu_di_%d'%i, nn.ReLU() )
        self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )
    def forward(self, in_tensor):
        return self.gate_s( in_tensor ).expand_as(in_tensor)
class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)
    def forward(self,in_tensor):
        att = 1 + F.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
        return att * in_tensor


#*SMRLayer
class SRMLayer(nn.Module):
    def __init__(self, channel):
        super(SRMLayer, self).__init__()

        self.cfc = Parameter(torch.Tensor(channel, 2))
        self.cfc.data.fill_(0)

        self.bn = nn.BatchNorm2d(channel)
        self.activation = nn.Sigmoid()

        setattr(self.cfc, 'srm_param', True)
        setattr(self.bn.weight, 'srm_param', True)
        setattr(self.bn.bias, 'srm_param', True)

    def _style_pooling(self, x, eps=1e-5):
        N, C, _, _ = x.size()

        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()

        t = torch.cat((channel_mean, channel_std), dim=2)
        return t 
    
    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None] # B x C x 1 x 1

        z_hat = self.bn(z)
        g = self.activation(z_hat)

        return g

    def forward(self, x):
        # B x C x 2
        t = self._style_pooling(x)

        # B x C x 1 x 1
        g = self._style_integration(t)

        return x * g


# *    channel代表in_channels  # *sSE 
class ChSqueezeAndSpExcitation(nn.Module):
    def __init__(self, channel):
        super(ChSqueezeAndSpExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = self.fc(x)
        y=weighting*x
        return y
 
 
   # *cSE  
class SpSqueezeAndChExcitation(nn.Module):
    def __init__(self, channel):
        super(SpSqueezeAndChExcitation, self).__init__()
        self.fc = nn.Sequential(
            #第一次全连接，降低维度
            nn.Conv2d(channel, channel // 2, kernel_size=1, bias=False),
            #第二次全连接，恢复维度
            nn.Conv2d(channel // 2, channel, kernel_size=1, bias=False), 
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y   
    
 #*scSE
class scSE(nn.Module):
    def __init__(self, channel):
        super(scSE,self).__init__()
        self.sSE=ChSqueezeAndSpExcitation(channel)
        self.cSE=SpSqueezeAndChExcitation(channel)
        
    def forward(self,x):
        y = torch.max(self.cSE(x), self.sSE(x))
        return y

# *ECA
class ESqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(ESqueezeAndExcitation, self).__init__()
        # self.fc = nn.Sequential(
        #     nn.Conv1d(1, 1 kernel_size=3,padding=(3 - 1) // 2, bias=False),
        #     activation,
        #     nn.Sigmoid()
        # )
        
        # *kernel_size=3,5,7,9,11  resnet越小，ksize越大
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3- 1) // 2, bias=False) 
        self.activation=activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # # feature descriptor on the global spatial information
        # y = F.adaptive_avg_pool2d(x, 1)
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y=self.activation(y)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# *SPA_124
# class  SPALayer(nn.Module):
#     def __init__(self,channel, reduction=16,activation=nn.ReLU(inplace=True)):
#         super(SPALayer, self).__init__()
#         self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
#         self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
#         self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
#         self.weight = Parameter(torch.ones(1, 3, 1, 1, 1))
#         self.transform = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, bias=False),
#             nn.BatchNorm2d(channel // reduction),
#             activation,
#             nn.Conv2d(channel // reduction, channel, 1, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y1 = self.avg_pool1(x)
#         y2 = self.avg_pool2(x)
#         y4 = self.avg_pool4(x)
#         y = torch.cat(
#             [y4.unsqueeze(dim=1),
#              F.interpolate(y2, scale_factor=2).unsqueeze(dim=1),
#              F.interpolate(y1, scale_factor=4).unsqueeze(dim=1)],
#             dim=1
#         )
#         y = (y * self.weight).sum(dim=1, keepdim=False)
#         y = self.transform(y)
#         y = F.interpolate(y, size=x.size()[2:])

#         return x * y
    
# *SPA_B
class SPABLayer(nn.Module):
    def __init__(self, inchannel,channel, reduction=16,activation=nn.ReLU(inplace=True)):
        super(SPABLayer, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.weight = Parameter(torch.ones(1,3,1,1,1))
        self.transform = nn.Sequential(
            nn.Conv2d(inchannel, channel//reduction,1,bias=False),
            nn.BatchNorm2d(channel//reduction),
            activation,
            nn.Conv2d(channel//reduction, channel, 1,bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c,_, _ = x.size()
        y1 = self.avg_pool1(x)
        y2 = self.avg_pool2(x)
        y4 = self.avg_pool4(x)
        y = torch.cat(
            [y4.unsqueeze(dim=1),
             F.interpolate(y2,scale_factor=2).unsqueeze(dim=1),
             F.interpolate(y1,scale_factor=4).unsqueeze(dim=1)],
            dim=1
        )
        y = (y*self.weight).sum(dim=1,keepdim=False)
        y = self.transform(y)

        return y

# *SPA_C
class SPACLayer(nn.Module):
    def __init__(self, inchannel,channel, reduction=16,activation=nn.ReLU(inplace=True)):
        super(SPACLayer, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.weight = Parameter(torch.ones(1,3,1,1,1))
        if inchannel !=channel:
            self.matcher = nn.Sequential(
                nn.Conv2d(inchannel, channel//reduction,1,bias=False),
                nn.BatchNorm2d(channel//reduction),
                activation,
                nn.Conv2d(channel//reduction, channel, 1,bias=False),
                nn.BatchNorm2d(channel)
            )
        self.transform = nn.Sequential(
            nn.Conv2d(channel, channel//reduction,1,bias=False),
            nn.BatchNorm2d(channel//reduction),
            activation,
            nn.Conv2d(channel//reduction, channel, 1,bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.matcher(x) if hasattr(self, 'matcher') else x
        b, c,_, _ = x.size()
        y1 = self.avg_pool1(x)
        y2 = self.avg_pool2(x)
        y4 = self.avg_pool4(x)
        y = torch.cat(
            [y4.unsqueeze(dim=1),
             F.interpolate(y2,scale_factor=2).unsqueeze(dim=1),
             F.interpolate(y1,scale_factor=4).unsqueeze(dim=1)],
            dim=1
        )
        y = (y*self.weight).sum(dim=1,keepdim=False)
        y = self.transform(y)

        return y

# # *SPA_147
class SPALayer2(nn.Module):
    def __init__(self,channel, reduction=16,activation=nn.ReLU(inplace=True)):
        super(SPALayer2, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.avg_pool7 = nn.AdaptiveAvgPool2d(7)
        self.weight = Parameter(torch.ones(1, 3, 1, 1, 1))
        self.transform = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.BatchNorm2d(channel // reduction),
            activation,
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool1(x)
        y2 = self.avg_pool4(x)
        y4 = self.avg_pool7(x)
        y = torch.cat(
            [y4.unsqueeze(dim=1),
             F.interpolate(y2, size=[7,7]).unsqueeze(dim=1),
             F.interpolate(y1, size=[7,7]).unsqueeze(dim=1)],
            dim=1
        )
        y = (y * self.weight).sum(dim=1, keepdim=False)
        y = self.transform(y)
        y = F.interpolate(y, size=x.size()[2:])

        return x * y, y


# # *SPA_147
class SPALayer(nn.Module):
    def __init__(self,channel, reduction=16,activation=nn.ReLU(inplace=True)):
        super(SPALayer, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.avg_pool7 = nn.AdaptiveAvgPool2d(7)
        self.weight = Parameter(torch.ones(1, 3, 1, 1, 1))
        self.transform = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.BatchNorm2d(channel // reduction),
            activation,
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool1(x)
        y2 = self.avg_pool4(x)
        y4 = self.avg_pool7(x)
        y = torch.cat(
            [y4.unsqueeze(dim=1),
             F.interpolate(y2, size=[7,7]).unsqueeze(dim=1),
             F.interpolate(y1, size=[7,7]).unsqueeze(dim=1)],
            dim=1
        )
        y = (y * self.weight).sum(dim=1, keepdim=False)
        y = self.transform(y)
        y = F.interpolate(y, size=x.size()[2:])

        return x * y
 
#  #*SPAD147
# class SPADLayer(nn.Module):
#     def __init__(self,channel, reduction=16,activation=nn.ReLU(inplace=True),k_size=9):
#         super(SPADLayer, self).__init__()
#         self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
#         self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
#         self.avg_pool7 = nn.AdaptiveAvgPool2d(7)
#         self.weight = Parameter(torch.ones(1, 3,  1, 1))
#         self.conv = nn.Conv2d(1, 256, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.transform = nn.Sequential(
#             # nn.Conv2d(channel, channel // reduction, 1, bias=False),
#             # nn.BatchNorm2d(channel // reduction),
#             # activation,
#             # nn.Conv2d(channel // reduction, channel, 1, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y1 = self.avg_pool1(x)
#         y2 = self.avg_pool4(x)
#         y4 = self.avg_pool7(x)
#         y = torch.cat(
#             [y4.unsqueeze(dim=1),
#              F.interpolate(y2, size=[7,7]).unsqueeze(dim=1),
#              F.interpolate(y1, size=[7,7]).unsqueeze(dim=1)],
#             dim=1
#         )
    
#         y = y.sum(dim=1, keepdim=False)
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         y = self.transform(y)
#         # y = F.interpolate(y, size=x.size()[2:])

#         return x * y.expand_as(x)
 
#  # *SPA_1247   
# class SPALayer(nn.Module):
#     def __init__(self,channel, reduction=16,activation=nn.ReLU(inplace=True)):
#         super(SPALayer, self).__init__()
#         self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
#         self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
#         self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
#         self.avg_pool7 = nn.AdaptiveAvgPool2d(7)

#         self.weight = Parameter(torch.ones(1, 4, 1, 1, 1))
#         self.transform = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, bias=False),
#             nn.BatchNorm2d(channel // reduction),
#             activation,
#             nn.Conv2d(channel // reduction, channel, 1, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y1 = self.avg_pool1(x)
#         y2 = self.avg_pool2(x)
#         y4 = self.avg_pool4(x)
#         y7 = self.avg_pool7(x)
#         y = torch.cat(
#             [y7.unsqueeze(dim=1),
#              F.interpolate(y4, size=[7, 7]).unsqueeze(dim=1),
#              F.interpolate(y2, size=[7, 7]).unsqueeze(dim=1),
#              F.interpolate(y1, size=[7, 7]).unsqueeze(dim=1)],
#             dim=1
#         )
#         y = (y * self.weight).sum(dim=1, keepdim=False)
#         y = self.transform(y)
#         y = F.interpolate(y, size=x.size()[2:])

#         return x * y
    
# *SPA_12
# class SPALayer(nn.Module):
#     def __init__(self,channel, reduction=16,activation=nn.ReLU(inplace=True)):
#         super(SPALayer, self).__init__()
#         self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
#         self.avg_pool2 = nn.AdaptiveAvgPool2d(2)

#         self.weight = Parameter(torch.ones(1, 2, 1, 1, 1))
#         self.transform = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, bias=False),
#             nn.BatchNorm2d(channel // reduction),
#             activation,
#             nn.Conv2d(channel // reduction, channel, 1, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y1 = self.avg_pool1(x)
#         y2 = self.avg_pool2(x)
#         y = torch.cat(
#             [y2.unsqueeze(dim=1),
#              F.interpolate(y1, size=[2,2]).unsqueeze(dim=1)],
#             dim=1
#         )
#         y = (y * self.weight).sum(dim=1, keepdim=False)
#         y = self.transform(y)
#         y = F.interpolate(y, size=x.size()[2:])

#         return x * y
  

class Swish(nn.Module):
    def forward(self, x):
        return swish(x)


def swish(x):
    return x * torch.sigmoid(x)


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.
