import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class EnhancementMetrics:
    """
    用于评估图像增强效果，包括 PSNR、SSIM 和 MSE。
    适用于经过 ToTensor() 归一化的图像，像素范围是 [0, 1]，然后将其转化为 [0, 255] 来计算指标。
    """
    def __init__(self):
        self.psnr_values = []
        self.ssim_values = []
        self.mse_values = []

    def update(self, preds, gts):
        """
        更新指标计算。
        preds: 预测结果，形状为 [B, C, H, W]，范围 [0, 1]。
        gts: 真实图像，形状同 preds，范围应一致。
        """
        # 确保 preds 和 gts 在 [0, 1] 范围内
        preds = np.clip(preds, 0, 1)
        gts = np.clip(gts, 0, 1)
        # 批量处理
        for pred, gt in zip(preds, gts):
            # 将图像的维度从 [C, H, W] 转为 [H, W, C]（skimage 需要的是这个格式）
            pred = np.transpose(pred, (1, 2, 0))  # 从 (C, H, W) 转为 (H, W, C)
            gt = np.transpose(gt, (1, 2, 0))
            # print(pred.shape,gt.shape)

            # 将 [0, 1] 范围的图像转换为 [0, 255] 范围，转换为整数类型
            pred = (pred * 255).astype(np.uint8)
            gt = (gt * 255).astype(np.uint8)

            # 计算 PSNR
            psnr_value = psnr(gt, pred, data_range=255)  # 对于 [0, 255] 范围的图像，data_range=255
            self.psnr_values.append(psnr_value)

            # 计算 SSIM
            ssim_value = ssim(gt, pred, multichannel=True, data_range=255)  # 同样 data_range=255
            self.ssim_values.append(ssim_value)

            # 计算 MSE
            mse_value = np.mean((gt - pred) ** 2)
            self.mse_values.append(mse_value)

    def get_results(self):
        """
        获取平均 PSNR、SSIM 和 MSE。
        """
        mean_psnr = np.mean(self.psnr_values) if self.psnr_values else 0
        mean_ssim = np.mean(self.ssim_values) if self.ssim_values else 0
        mean_mse = np.mean(self.mse_values) if self.mse_values else 0

        return {
            "mean_PSNR": mean_psnr,
            "mean_SSIM": mean_ssim,
            "mean_MSE": mean_mse
        }

    def reset(self):
        """重置指标记录。"""
        self.psnr_values = []
        self.ssim_values = []
        self.mse_values = []