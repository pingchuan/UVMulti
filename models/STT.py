import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm3D(nn.Module):
    def __init__(self, channels):
        super(LayerNorm3D, self).__init__()
        # LayerNorm expects the shape to be [batch_size, num_features, ...], we apply it on channels (C)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x is expected to have shape [batch_size, channels, height, width]
        batch_size, channels, height, width = x.size()

        # Flatten the spatial dimensions (height, width) so LayerNorm can only apply on the channel dimension
        x = x.view(batch_size, channels, -1)  # Flatten to shape: [batch_size, channels, height * width]

        # Apply LayerNorm along the channels dimension
        x = self.norm(x)

        # Reshape back to the original spatial shape
        x = x.view(batch_size, channels, height, width)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return x


class MixFFN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(MixFFN, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)  # Depthwise 3x3 convolution
        self.mlp = MLP(in_channels, hidden_channels, in_channels)  # MLP layer in-between

    def forward(self, x):
        # Apply the 3x3 convolution followed by MLP
        x = self.conv(x)

        x = self.mlp(x.view(x.size(0), -1))  # Flatten for MLP
        return x.view_as(x)


# Define LiteMLA as provided earlier
class LiteMLA(nn.Module):
    def __init__(self, in_channels, out_channels, heads_ratio=1.0, dim=8, scales=(3,5), eps=1.0e-15):
        super(LiteMLA, self).__init__()
        self.eps = eps
        heads = int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        self.dim = dim
        self.qkv = nn.Conv2d(
            in_channels,
            3 * total_dim,
            1,
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=scale // 2,
                        groups=3 * total_dim,
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=heads),
                )
                for scale in scales
            ]
        )

        self.proj = nn.Conv2d(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
        )

    def forward(self, x):
        # Generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        qkv = torch.cat(multi_scale_qkv, dim=1)

        H, W = qkv.size()[-2:]
        out = self.relu_linear_att(qkv).to(qkv.dtype)
        out = self.proj(out)

        return out

    def relu_linear_att(self, qkv: torch.Tensor):
        B, _, H, W = list(qkv.size())

        qkv = torch.reshape(qkv, (B, -1, 3 * self.dim, H * W))
        q, k, v = (
            qkv[:, :, 0: self.dim],
            qkv[:, :, self.dim: 2 * self.dim],
            qkv[:, :, 2 * self.dim:],
        )

        q = F.relu(q)
        k = F.relu(k)

        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        vk = torch.matmul(v, trans_k)
        out = torch.matmul(vk, q)
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        out = torch.reshape(out, (B, -1, H, W))
        return out


# Define the STTBlock with LiteMLA
class STTBlock(nn.Module):
    def __init__(self, in_channels,  hidden_channels):
        super(STTBlock, self).__init__()
        # Replacing WM-SA with LiteMLA
        self.context_module = LiteMLA(in_channels=in_channels, out_channels=in_channels, heads_ratio=1.0, dim=8,
                                      scales=(3,5))
        self.ffn = MixFFN(in_channels, hidden_channels)
        self.norm1 = LayerNorm3D(in_channels)
        self.norm2 = LayerNorm3D(in_channels)

    def forward(self, x):
        # Apply LiteMLA, followed by FFN and layer norm
        x_res = x
        #x = self.norm1(x)
        x = self.context_module(x)  # Replacing WM-SA with LiteMLA
        x = x_res + x  # Skip connection

        x_res = x
        #x = self.norm2(x)
        #x = self.ffn(x)
        x = x_res + x  # Skip connection

        return x


class MultiFrameFeatureFusion(nn.Module):
    def __init__(self, in_channels_list, window_size=7, frames=3):
        super(MultiFrameFeatureFusion, self).__init__()

        # Define individual STTBlocks for each scale (layer1, layer2, layer3, layer4)
        self.stt_blocks = nn.ModuleList([
            STTBlock(in_channels=in_channels, hidden_channels=in_channels)
            for in_channels in in_channels_list  # Different in_channels for each scale
        ])
        self.fc = nn.Conv2d(
            in_channels=sum(in_channels_list),  # The total number of channels after concatenation
            out_channels=in_channels_list[0],  # Match the channels to the first scale (current frame)
            kernel_size=1
        )

    def forward(self, current_frame_feats, prev_frame_feats, prev2_frame_feats):
        """
        :param current_frame_feats: List of features from current frame [layer1, layer2, layer3, layer4]
        :param prev_frame_feats: List of features from previous frame [layer1, layer2, layer3, layer4]
        :param prev2_frame_feats: List of features from second previous frame [layer1, layer2, layer3, layer4]
        :return: Concatenated and fused feature map
        """

        # Process each scale of the current, previous, and second previous frames
        fused_feats = []
        target_size = current_frame_feats[1].shape[2:]
        for scale_idx in range(1,5):

            current_feat = current_frame_feats[scale_idx]
            prev_feat = prev_frame_feats[scale_idx]
            prev2_feat = prev2_frame_feats[scale_idx]

            # Concatenate the current, previous, and second previous frame features along the channel dimension
            # Shape: [batch_size, 3 * in_channels, H, W]
            #concatenated_feat = torch.cat([current_feat, prev_feat, prev2_feat], dim=1)
            # 对应元素相加，而不是拼接
            concatenated_feat = current_feat + prev_feat + prev2_feat

            # Apply the corresponding STTBlock for the concatenated features
            fused_feat = self.stt_blocks[scale_idx-1](concatenated_feat)
            #fused_feat = F.interpolate(fused_feat, size=target_size, mode='bilinear', align_corners=False)
            # Append the fused feature for this scale
            fused_feats.append(fused_feat)
        #final_fused_feat = torch.cat(fused_feats, dim=1)  # Concatenate along the channel dimension

        # Pass the concatenated features through the final convolutional layer to reduce channels
        #final_fused_feat = self.fc(final_fused_feat)

        return fused_feats  # A list of 4 fused features, one for each scale

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


class SegDecoder(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super(SegDecoder, self).__init__()
        # Decoder
        self.multiframes = TAU(in_channels=64*3, t_dim=3)
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


    def forward(self, x,x1,x2):
        self.outputs = {}
        e1, e2, e3, e4, e5 = x
        #print(e1.shape,e2.shape,e3.shape,e4.shape,e5.shape)
        e1_1, _, _, _, _ = x1
        e1_2, _, _, _, _ = x2
        #print(e1.shape,e2.shape)
        result = torch.cat((e1, e1_1, e1_2), dim=1)
        m_f = self.multiframes(result)
        #print(m_f[0].shape,m_f[1].shape)
        d5 = self.decoder5(e5)  # 14
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))  # 28
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))  # 56
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))  # 128
        d1 = self.decoder1(torch.cat((d2, e1+m_f), dim=1))  # 224*224*64

        out1 = self.outconv(d1)  # 原始是d1
        self.outputs["seg"] = out1
        return self.outputs

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



class SegDecoder50(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super(SegDecoder50, self).__init__()
        # Decoder
        features = int(2048)

        self.conv2 = nn.Conv2d(2048, features, kernel_size=1, stride=1, padding=1)
        # for res50
        self.up1 = UpSampleBN(skip_input=features // 1 + 1024, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + 512, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + 256, output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + 64, output_features=features // 16)

        # self.up5 = UpSampleBN(skip_input=features // 16 + 3, output_features=features//16)
        #self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)
        self.outconv = nn.Sequential(
            ConvBlock(128, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, num_classes, 1),
        )


    def forward(self, x,x1,x2):
        self.outputs = {}
        x_block0, x_block1, x_block2, x_block3, x_block4 = x

        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        # x_d5 = self.up5(x_d4, features[0])
        # out = self.conv3(x_d5)
        #out = self.conv3(x_d4)

        out1 = self.outconv(x_d4)  # 原始是d1
        self.outputs["seg"] = out1
        return self.outputs

class TAU(nn.Module):
    def __init__(self, in_channels, t_dim):
        """
        :param in_channels: 输入特征图的通道数
        :param t_dim: 时间维度 T
        :param h: 高度 H
        :param w: 宽度 W
        """
        super(TAU, self).__init__()

        # 1x1 卷积，用于将通道数降低，输出的通道数为 in_channels
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # DW Conv（Depthwise Conv）和 DW-D Conv（Dilated Depthwise Conv）
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.dw_dconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2, groups=in_channels)
        self.t_dim = t_dim
        self.in_channels = in_channels
        # 动态注意力模块（Dynamical Attention）
        self.fc = nn.Linear(in_channels, in_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 初始化参数
        self.init_weights()

    def init_weights(self):
        # 初始化卷积层和全连接层权重
        nn.init.kaiming_normal_(self.conv1x1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.dw_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.dw_dconv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """
        :param x: 输入特征图 [B, T * C, H, W]
        :return: 输出特征图 [B, T * C, H, W]
        """
        # 1. 计算 statical attention (SA)
        B,_,H,W = x.size()
        #sa = self.dw_dconv(self.dw_conv(x))  # 深度卷积和扩张卷积组合
        sa = self.conv1x1(x)  # 1x1 卷积

        # 2. 计算 dynamical attention (DA)
        # 平均池化，得到 [B, C, 1, 1]
        avg_pool = self.avg_pool(x)
        avg_pool = avg_pool.view(avg_pool.size(0), -1)  # 展平为 [B, T * C]
        da = self.fc(avg_pool)  # 全连接层，得到通道注意力权重
        da = da.view(da.size(0), da.size(1), 1, 1)  # 转换为 [B, T * C, 1, 1]

        # 3. 合并 statical 和 dynamical 注意力
        # Kronecker product: 扩展 SA 和 DA 到相同形状
        sa = sa * da  # 这里是 Hadamard product，按元素相乘
        sa = sa.view(B, self.t_dim, self.in_channels//self.t_dim, H, W)  # (B, t_dim, in_channels, H, W)
        sa = sa.sum(dim=1)  # 在 t_dim 维度上求和，得到 (B, in_channels, H, W)

        # 进行 Hadamard product

        return sa

# Example Usage
if __name__ == "__main__":
    batch_size = 2
    in_channels = 64
    num_heads = 8
    hidden_channels = 128
    window_size = 7
    depth = 32
    height = 128
    width = 128

    # Create a random 3D input tensor
    x = torch.randn(batch_size, in_channels, depth, height, width)

    # Instantiate the STT block with LiteMLA
    stt_block = STTBlock(in_channels, num_heads, hidden_channels, window_size)

    # Forward pass
    out = stt_block(x)
    print(out.shape)  # Expected shape: [batch_size, in_channels, depth, height, width]
