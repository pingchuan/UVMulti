import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random




def mse_consistency_loss(pred_F, pred_G):
    """
    计算 MSE 一致性损失。

    参数:
    pred_F: 第一个模型的预测结果
    pred_G: 第二个模型的预测结果

    返回:
    损失值
    """
    # 计算均方误差
    loss = torch.mean((pred_F - pred_G) ** 2)
    return loss

class DiceLoss2_s(nn.Module):
    def __init__(self, smooth=1e-5):
        """
        初始化 Dice 损失函数。

        参数：
        - smooth (float): 避免除以零的平滑项。
        """
        super(DiceLoss2_s, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target, numclass):



        dice = 0.0
        for c in range(numclass):

            intersection = torch.sum(pred[:, c, :, :] * target)
            union = torch.sum(pred[:, c, :, :]) + torch.sum(target)
            dice += (2. * intersection + self.smooth) / (union + intersection + self.smooth)


        return 1 - (dice / numclass)


class FCDLoss(nn.Module):
    def __init__(self):
        super(FCDLoss, self).__init__()

    def forward(self, f1, f2):
        """
        f1: feature map from the first layer (after pointwise convolution), shape: (batch_size, C, H, W)
        f2: feature map from the second layer, shape: (batch_size, C, H2, W2)
        """
        # Ensure f1 and f2 are 4D tensors
        assert f1.ndimension() == 4 and f2.ndimension() == 4, 'Input tensors must be 4D'

        batch_size, C1, H1, W1 = f1.shape
        batch_size, C2, H2, W2 = f2.shape

        # 1. Check if the channel dimensions of f1 and f2 are already aligned
        assert C1 == C2, 'Channels of f1 and f2 must be the same'

        # 2. Downsample f2 to match the spatial size of f1 (H1, W1)
        # f2_downsampled = F.interpolate(f2, size=(H1, W1), mode='bilinear', align_corners=False)  # Shape: (batch_size, C, H1, W1)
        f2_downsampled = F.adaptive_avg_pool2d(f2, (H1, W1))
        # 3. Compute L2 loss (mean square error)
        loss = F.mse_loss(f1, f2_downsampled)

        return loss


class SCRLoss(nn.Module):
    def __init__(self):
        super(SCRLoss, self).__init__()

    def forward(self, f1, f2):


        assert f1.ndimension() == 4 and f2.ndimension() == 4, 'Input tensors must be 4D'

        batch_size, C1, H1, W1 = f1.shape
        batch_size, C2, H2, W2 = f2.shape

        # 1. Randomly select C2 channels from f1
        random_channels = torch.randint(0, C1, (batch_size, C2), device=f1.device)  # Shape: (batch_size, C2)
        selected_f1 = torch.stack([f1[i, random_channels[i], :, :] for i in range(batch_size)],
                                  dim=0)  # Shape: (batch_size, C2, H1, W1)

        # 2. Apply average pooling to f2 to match the spatial size of f1
        f2_pooled = F.adaptive_avg_pool2d(f2, (H1, W1))  # Shape: (batch_size, C2, H1, W1)

        # 3. Compute L2 loss (mean square error)
        loss = F.mse_loss(selected_f1, f2_pooled)

        return loss

class BDiceLoss_sup(nn.Module):
    def __init__(self, dice_weight=1.0, numclass=6):
        super(BDiceLoss_sup, self).__init__()
        self.dice = DiceLoss2_s()
        self.dice_weight = dice_weight
        self.numclass = numclass

    def forward(self, pred, target):


        pred_softmax = F.softmax(pred, dim=1)


        bce_loss = F.cross_entropy(pred, target, reduction='none')


        bce_loss = bce_loss.sum() / (target.numel() + 1e-8)


        dice_loss = self.dice(pred_softmax, target, numclass=self.numclass)


        loss = 0.5*bce_loss + 0.5 * dice_loss

        return loss






