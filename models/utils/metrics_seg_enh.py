import torch
import numpy as np
import cv2
class StreamSegMetrics:
    """
    用于计算累积混淆矩阵的语义分割指标，包括 mIoU 和整体准确率。
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        # 初始化混淆矩阵
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, preds, gts):
        """
        更新混淆矩阵。
        preds: 预测结果，形状为 [B, H, W] 或 [H, W]
        gts: 真实标签，形状为 [B, H, W] 或 [H, W]
        """
        for pred, gt in zip(preds, gts):
            self.confusion_matrix += self._fast_hist(gt.flatten(), pred.flatten())

    def _fast_hist(self, label_true, label_pred):
        """
        计算单张图片的混淆矩阵。
        label_true: 展平后的 ground truth 标签。
        label_pred: 展平后的预测结果。
        """

        # 将标签调整为整数

        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def get_results(self):
        """
        基于混淆矩阵计算指标：
        - 每个类别的 IoU
        - 平均 IoU (mIoU)
        - 整体准确率 (Overall Accuracy)
        """
        hist = self.confusion_matrix
        # Overall Accuracy
        overall_acc = np.diag(hist).sum() / hist.sum()

        # Per-class IoU
        TP = np.diag(hist)
        FP = hist.sum(axis=0) - TP
        FN = hist.sum(axis=1) - TP
        IoU = TP / (TP + FP + FN + 1e-8)

        # Mean IoU (忽略背景类别)
        mean_IoU = np.nanmean(IoU[1:])  # 忽略第0类

        # Per-class accuracy
        acc_cls = TP / (hist.sum(axis=1) + 1e-8)
        mean_acc_cls = np.nanmean(acc_cls[1:])  # 忽略第0类
        IoU_per_class = {f'IoU_class_{cls}': IoU[cls].item() for cls in range(1, self.num_classes)}

        # 返回每个前景类别的IoU、平均IoU和整体准确率
        return {
            #**IoU_per_class,  # 每个前景类别的IoU
            'mean_IoU': mean_IoU.item(),  # 所有前景类别的平均IoU
            'mean_ACC': mean_acc_cls,  # 整体准确率，包括背景
            'ACC_overall': overall_acc
        }




    def reset(self):
        """重置混淆矩阵。"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

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

            gt_height, gt_width = gt.shape[:2]

            # 预测图像需要进行resize，以匹配真实图像的尺寸
            pred = cv2.resize(pred, (gt_width, gt_height), interpolation=cv2.INTER_LANCZOS4)
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

            print("psnr:", psnr_value, ssim_value, mse_value)

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


class EnhancementMetrics_255:
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
            ssim_value = ssim(gt, pred, multichannel=True, data_range=255, range=255)  # 同样 data_range=255
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