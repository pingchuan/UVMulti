# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/depth_decoder.py
#
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


from torchvision.models.segmentation.deeplabv3 import ASPPConv, ASPPPooling
def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")
class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels, bn=False, dropout=0.0):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            Conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.ELU(inplace=True),
            # Pay attention: 2d version of dropout is used
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        out = self.block(x)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, aspp_pooling=True, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        for r in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, r))
        if aspp_pooling:
            modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d((1 + int(aspp_pooling) + len(atrous_rates)) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DepthDecoder_f(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder_f, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs

class DepthDecoder_aspp(nn.Module):
    first_iter = True

    def __init__(self, num_ch_enc, scales, max_scale_size, num_output_channels=1, use_skips=True,
                 intermediate_aspp=False, aspp_rates=[6, 12, 18], num_ch_dec=[16, 32, 64, 128, 256],
                 n_upconv=4, batch_norm=False, dropout=0.0, n_project_skip_ch=-1,
                 aspp_pooling=True):
        super(DepthDecoder_aspp, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.enable_disparity = True
        self.max_scale_size = np.asarray(max_scale_size)

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array(num_ch_dec)
        self.n_upconv = n_upconv

        # decoder
        self.convs = OrderedDict()
        for i in range(self.n_upconv, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == self.n_upconv else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            if i == self.n_upconv and intermediate_aspp:
                self.convs[("upconv", i, 0)] = ASPP(num_ch_in, aspp_rates, aspp_pooling, num_ch_out)
            else:
                self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out, bn=batch_norm, dropout=dropout)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                if n_project_skip_ch == -1:
                    num_ch_in += self.num_ch_enc[i - 1]
                    self.convs[("skip_proj", i)] = nn.Identity()
                else:
                    num_ch_in += n_project_skip_ch
                    self.convs[("skip_proj", i)] = nn.Sequential(
                        nn.Conv2d(self.num_ch_enc[i - 1], n_project_skip_ch, kernel_size=1),
                        nn.BatchNorm2d(n_project_skip_ch),
                        nn.ReLU(inplace=True)
                    )
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out, bn=batch_norm, dropout=dropout)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, x=None, exec_layer=None):
        self.outputs = {}

        # decoder
        if x is None:
            x = input_features[-1]
        if exec_layer is None:
            exec_layer = "all"
        if DepthDecoder.first_iter:
            print(f"bottleneck shape {x.shape}")
        for i in range(self.n_upconv, -1, -1):
            if exec_layer != "all" and i not in exec_layer:
                continue
            x = self.convs[("upconv", i, 0)](x)
            # if i == self.n_upconv:
            #     self.outputs["aspp"] = x
            if DepthDecoder.first_iter:
                print(f"upconv{i}-0 shape: {x.shape}")
            if x.shape[-1] < input_features[i - 1].shape[-1] or i == 0:
                x = [upsample(x)]
            else:
                x = [x]
            if self.use_skips and i > 0:
                projected_features = self.convs[("skip_proj", i)](input_features[i - 1])
                x += [projected_features]
            x = torch.cat(x, 1)
            if DepthDecoder.first_iter:
                print(f"concatenated features shape: {x.shape}")
            x = self.convs[("upconv", i, 1)](x)
            self.outputs[("upconv", i)] = x
            if DepthDecoder.first_iter:
                print(f"upconv{i}-1 shape: {x.shape}")
            if i in self.scales and self.enable_disparity:
                size = self.max_scale_size // (2 ** i)
                disp_out = self.sigmoid(self.convs[("dispconv", i)](x))
                if DepthDecoder.first_iter:
                    print(f"disp{i} shape: {disp_out.shape}, expected {size}")
                self.outputs[("disp", i)] = disp_out
            if i == 0:
                DepthDecoder.first_iter = False

        return self.outputs
