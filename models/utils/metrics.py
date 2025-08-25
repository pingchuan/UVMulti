import torch
import numpy as np


def evaluate_multiclass(pred, gt, num_classes):
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    # 进行 softmax 转换
    pred_probs = torch.softmax(pred, dim=1)
    pred_classes = pred_probs.argmax(dim=1)

    # 转换 ground truth 为类索引
    gt_classes = gt.argmax(dim=1)

    # 计算 TP, FP, TN, FN
    TP = torch.zeros(num_classes).float().to(pred.device)
    FP = torch.zeros(num_classes).float().to(pred.device)
    TN = torch.zeros(num_classes).float().to(pred.device)
    FN = torch.zeros(num_classes).float().to(pred.device)

    for cls in range(num_classes):
        TP[cls] = ((pred_classes == cls) & (gt_classes == cls)).sum().float()
        FP[cls] = ((pred_classes == cls) & (gt_classes != cls)).sum().float()
        TN[cls] = ((pred_classes != cls) & (gt_classes != cls)).sum().float()
        FN[cls] = ((pred_classes != cls) & (gt_classes == cls)).sum().float()

    # 计算各类指标
    Precision = TP / (TP + FP + 1e-8)
    Recall = TP / (TP + FN + 1e-8)
    F1 = 2 * Precision * Recall / (Precision + Recall + 1e-8)

    # 全局指标
    total_TP = TP.sum()
    total_FP = FP.sum()
    total_TN = TN.sum()
    total_FN = FN.sum()

    ACC_overall = (total_TP + total_TN) / (total_TP + total_FP + total_FN + total_TN + 1e-8)

    # 计算 Dice Score
    dice_scores = 2 * TP / (2 * TP + FP + FN + 1e-8)
    mean_dice_score = dice_scores.mean()

    # 计算 IoU
    IoU = TP / (TP + FP + FN + 1e-8)
    mIoU = IoU.mean()

    return {
        'precision': Precision.mean().item(),
        'recall': Recall.mean().item(),
        'F1': F1.mean().item(),
        'ACC_overall': ACC_overall.item(),
        'mDice': mean_dice_score.item(),
        'mIoU': mIoU.item(),
    }

class Metrics(object):
    def __init__(self, metrics_list):
        self.metrics = {metric: [] for metric in metrics_list}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.metrics:
                if isinstance(v, torch.Tensor):
                    v = v.item()
                self.metrics[k].append(v)

    def mean(self):
        mean_metrics = {k: np.mean(v) for k, v in self.metrics.items()}
        return mean_metrics

    def clean(self):
        for k in self.metrics.keys():
            self.metrics[k].clear()
