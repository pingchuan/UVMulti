import torch
import numpy as np
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
            **IoU_per_class,  # 每个前景类别的IoU
            'mean_IoU': mean_IoU.item(),  # 所有前景类别的平均IoU
            'ACC_overall': overall_acc,
            'mean_ACC': mean_acc_cls  # 整体准确率，包括背景
        }




    def reset(self):
        """重置混淆矩阵。"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)