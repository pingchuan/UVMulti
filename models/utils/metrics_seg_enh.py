import torch
import numpy as np
import cv2
class StreamSegMetrics:

    def __init__(self, num_classes):
        self.num_classes = num_classes

        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, preds, gts):


        for pred, gt in zip(preds, gts):
            self.confusion_matrix += self._fast_hist(gt.flatten(), pred.flatten())

    def _fast_hist(self, label_true, label_pred):

        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def get_results(self):

        hist = self.confusion_matrix
        # Overall Accuracy
        overall_acc = np.diag(hist).sum() / hist.sum()

        # Per-class IoU
        TP = np.diag(hist)
        FP = hist.sum(axis=0) - TP
        FN = hist.sum(axis=1) - TP
        IoU = TP / (TP + FP + FN + 1e-8)


        mean_IoU = np.nanmean(IoU[1:])  # 忽略第0类

        # Per-class accuracy
        acc_cls = TP / (hist.sum(axis=1) + 1e-8)
        mean_acc_cls = np.nanmean(acc_cls[1:])  # 忽略第0类
        IoU_per_class = {f'IoU_class_{cls}': IoU[cls].item() for cls in range(1, self.num_classes)}


        return {
            #**IoU_per_class,
            'mean_IoU': mean_IoU.item(),
            'mean_ACC': mean_acc_cls,
            'ACC_overall': overall_acc
        }




    def reset(self):
        """重置混淆矩阵。"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class EnhancementMetrics:

    def __init__(self):
        self.psnr_values = []
        self.ssim_values = []
        self.mse_values = []

    def update(self, preds, gts):

        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(gts, torch.Tensor):
            gts = gts.detach().cpu().numpy()


        preds = np.clip(preds, 0, 1)
        gts = np.clip(gts, 0, 1)


        for pred, gt in zip(preds, gts):

            gt_height, gt_width = gt.shape[:2]


            pred = cv2.resize(pred, (gt_width, gt_height), interpolation=cv2.INTER_LANCZOS4)

            pred = np.transpose(pred, (1, 2, 0))
            gt = np.transpose(gt, (1, 2, 0))

            psnr_value = psnr(gt, pred,data_range=1.0)
            self.psnr_values.append(psnr_value)


            ssim_value = ssim(gt, pred, multichannel=True, data_range=1.0, range=1.0)
            self.ssim_values.append(ssim_value)


            mse_value = np.mean((gt - pred) ** 2)
            self.mse_values.append(mse_value)

            print("psnr:", psnr_value, ssim_value, mse_value)

    def get_results(self):

        mean_psnr = np.mean(self.psnr_values)
        mean_ssim = np.mean(self.ssim_values)
        mean_mse = np.mean(self.mse_values)

        return {
            "mean_PSNR": mean_psnr,
            "mean_SSIM": mean_ssim,
            "mean_MSE": mean_mse
        }

    def reset(self):

        self.psnr_values = []
        self.ssim_values = []
        self.mse_values = []


class EnhancementMetrics_255:

    def __init__(self):
        self.psnr_values = []
        self.ssim_values = []
        self.mse_values = []

    def update(self, preds, gts):

        preds = np.clip(preds, 0, 1)
        gts = np.clip(gts, 0, 1)

        for pred, gt in zip(preds, gts):

            pred = np.transpose(pred, (1, 2, 0))
            gt = np.transpose(gt, (1, 2, 0))

            pred = (pred * 255).astype(np.uint8)
            gt = (gt * 255).astype(np.uint8)


            psnr_value = psnr(gt, pred, data_range=255)
            self.psnr_values.append(psnr_value)


            ssim_value = ssim(gt, pred, multichannel=True, data_range=255, range=255)
            self.ssim_values.append(ssim_value)


            mse_value = np.mean((gt - pred) ** 2)
            self.mse_values.append(mse_value)


    def get_results(self):

        mean_psnr = np.mean(self.psnr_values) if self.psnr_values else 0
        mean_ssim = np.mean(self.ssim_values) if self.ssim_values else 0
        mean_mse = np.mean(self.mse_values) if self.mse_values else 0

        return {
            "mean_PSNR": mean_psnr,
            "mean_SSIM": mean_ssim,
            "mean_MSE": mean_mse
        }

    def reset(self):
       
        self.psnr_values = []
        self.ssim_values = []
        self.mse_values = []