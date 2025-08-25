import torch
import numpy as np
def evaluate_multiclass_ignore_background(pred, gt, num_classes):
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    # 进行 softmax 转换
    pred_probs = torch.softmax(pred, dim=1)
    pred_classes = pred_probs.argmax(dim=1).squeeze(0)

    # 转换 ground truth 为类索引
    gt_classes = gt.squeeze()


    # 计算 TP, FP, FN，仅针对非背景类别
    TP = torch.zeros(num_classes - 1).float().to(pred.device)  # 只为1到num_classes-1的类别
    FP = torch.zeros(num_classes - 1).float().to(pred.device)
    FN = torch.zeros(num_classes - 1).float().to(pred.device)

    for cls in range(1, num_classes):  # 从1开始，忽略类别0
        TP[cls - 1] = ((pred_classes == cls) & (gt_classes == cls)).sum().float()
        FP[cls - 1] = ((pred_classes == cls) & (gt_classes != cls)).sum().float()
        FN[cls - 1] = ((pred_classes != cls) & (gt_classes == cls)).sum().float()


    # 计算 TP, FP, TN, FN 总数
    total_TP = TP.sum().item()
    total_FP = FP.sum().item()
    total_FN = FN.sum().item()
    total_TN = ((pred_classes == 0) & (gt_classes == 0)).sum().item()  # 背景正确预测数

    # 计算每个类别的IoU
    IoU = TP / (TP + FP + FN + 1e-8)
    mean_IoU = IoU.mean()  # 所有前景类别的平均IoU

    # 计算整体准确率，包括所有类别
    total_preds = pred_classes.numel()  # 所有预测数
    ACC_overall = (total_TP + total_TN) / (total_preds + 1e-8)

    # 构建每个前景类别的IoU字典
    IoU_per_class = {f'IoU_class_{cls}': IoU[cls - 1].item() for cls in range(1, num_classes)}

    # 返回每个前景类别的IoU、平均IoU和整体准确率
    return {
        **IoU_per_class,                     # 每个前景类别的IoU
        'mean_IoU': mean_IoU.item(),        # 所有前景类别的平均IoU
        'ACC_overall': ACC_overall,   # 整体准确率，包括背景
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
