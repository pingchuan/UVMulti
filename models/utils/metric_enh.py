import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class EnhancementMetrics:
    """
    用于评估图像增强效果，包括 PSNR、SSIM 和 MSE。
    """
    def __init__(self):
        self.psnr_values = []
        self.ssim_values = []
        self.mse_values = []

    def update(self, preds, gts):
        """
        更新指标计算。
        preds: 预测结果，形状为 [B, H, W, C] 或 [H, W, C]，范围 0-1 或 0-255。
        gts: 真实图像，形状同 preds，范围应一致。
        """
        if isinstance(preds, torch.Tensor):  # 如果是 PyTorch 张量，转换为 NumPy
            preds = preds.detach().cpu().numpy()
        if isinstance(gts, torch.Tensor):
            gts = gts.detach().cpu().numpy()

        # 确保范围一致
        preds = np.clip(preds, 0, 1)
        gts = np.clip(gts, 0, 1)

        # 批量处理
        for pred, gt in zip(preds, gts):
            # 计算 PSNR
            pred = np.transpose(pred, (1, 2, 0))  # 从 (C, H, W) 转为 (H, W, C)
            gt = np.transpose(gt, (1, 2, 0))
            #print(pred.shape,gt.shape)
            psnr_value = psnr(gt, pred,data_range=1.0)
            self.psnr_values.append(psnr_value)

            # 计算 SSIMssim(im1, im2, multichannel=True)
            ssim_value = ssim(gt, pred, multichannel=True, data_range=1.0, range=1.0)
            self.ssim_values.append(ssim_value)

            # 计算 MSE
            mse_value = np.mean((gt - pred) ** 2)
            self.mse_values.append(mse_value)

    def get_results(self):
        """
        获取平均 PSNR、SSIM 和 MSE。
        """
        mean_psnr = np.mean(self.psnr_values)
        mean_ssim = np.mean(self.ssim_values)
        mean_mse = np.mean(self.mse_values)

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
