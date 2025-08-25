import os
import torch
from models.utils.transform import *
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
from PIL import ImageFilter
from PIL import Image
def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img
def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask

colors = [
        [0,   0,   0],    # 背景
        [128, 0,   0],    # relic
        [0,   128, 0],    # urchin
        [128, 128,   0],    # starfish
        [0, 0,   128],    # fish
        [128,   0,   128],  # reefs
        [0,   128,   128], #dead coral
    ]
excluded_videos = [
    "video1", "video13", "video101", "video118", "video119", "video124",
    "video129", "video132", "video137", "video14", "video25", "video38",
    "video42", "video43", "video51", "video66", "video84", "video86", "video92", "video98"
]
# KavSir-SEG Dataset
# KavSir-SEG Dataset

def compute_repeat_factors(class_frequencies, threshold=60):
    """
    根据类别频率计算重复因子
    Args:
        class_frequencies: 每个类别的频率
        threshold: 最小频率阈值

    Returns:
        repeat_factors: 每个类别的重复因子
    """
    repeat_factors = np.maximum(1, np.sqrt(threshold / class_frequencies))
    return repeat_factors


def compute_frame_weights(labeled_indices, repeat_factors, num_classes=7, color_to_label=None):
    """
    计算每一帧的权重，根据标签图像中的RGB值映射到类别，并根据类别的repeat_factors来计算帧的权重。

    参数:
    - labeled_indices: 一个包含元组的列表，每个元组由(_, frame, gt_path)组成
    - repeat_factors: 每个类别对应的重复因子列表
    - num_classes: 标签的类别数量，默认为7
    - color_to_label: RGB到类别的映射字典

    返回:
    - frame_weights: 包含每帧的权重的列表
    """
    frame_weights = []

    # 默认的 RGB 到类别的映射字典
    if color_to_label is None:
        color_to_label = {
            (0, 0, 0): 0,  # 背景
            (128, 0, 0): 1,  # relic
            (0, 128, 0): 2,  # urchin
            (128, 128, 0): 3,  # starfish
            (0, 0, 128): 4,  # fish
            (128, 0, 128): 5,  # reefs
            (0, 128, 128): 6,  # dead coral
        }

    for _, frame, gt_path in labeled_indices:
        # 加载地面真实标签图像，并转换为RGB格式
        label_path = os.path.join(gt_path, frame.replace('.jpg', '.png'))
        label = Image.open(label_path).convert('RGB')  # 转换为RGB模式
        label_array = np.array(label)

        # 计算帧的初始权重
        frame_factor = 1
        for color, label_value in color_to_label.items():
            # 创建掩码，匹配每个像素的RGB值
            mask = np.all(label_array == color, axis=-1)
            # 如果该颜色存在，则计算该帧的权重
            if np.any(mask):
                frame_factor = max(frame_factor, repeat_factors[label_value])

        # 将当前帧的权重添加到 frame_weights 列表中
        frame_weights.append(frame_factor)

    return frame_weights

class BenthicVideoDataset_SUP_RFS(Dataset):
    def __init__(self, root, data2_dir, mode='train', transform=None):
        super(BenthicVideoDataset_SUP_RFS, self).__init__()
        self.root = root
        self.mode = mode
        self.data_path = os.path.join(root, data2_dir)
        self.video_dirs = sorted(os.listdir(self.data_path))  # 读取视频文件夹
        self.labeled_indices = []
        self.seed = 100
        self.set_seed(self.seed)

        if self.mode == 'train':
            # 如果是 train 模式，只加载排除的视频
            self.video_dirs = [video for video in self.video_dirs if video in excluded_videos]
        # 遍历每个视频
        for video in self.video_dirs:

            video_path = os.path.join(self.data_path, video)
            gt_path = os.path.join(video_path, 'mask')  # 标注路径
            sequence_path = os.path.join(video_path, 'images')  # 图像路径

            frames = sorted(os.listdir(sequence_path))
            gt_frames = {f.replace('.png', '.jpg') for f in os.listdir(gt_path)}  # 获取标注帧集合，转换为 JPEG 名称

            # 遍历所有帧，生成带标注的样本
            for frame in frames:
                if frame in gt_frames:  # 检查当前帧是否有对应的标签
                    self.labeled_indices.append((sequence_path, frame, gt_path))  # 添加到带标注的列表

        # 随机打乱 labeled_indices
        random.shuffle(self.labeled_indices)

        if transform is None:
            if mode == 'train':
                transform = transforms.Compose([
                    Resize((320, 320)),  # 1280,720 四次降采样正好
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                    RandomRotation(90),
                ])
            elif mode == 'valid' or mode == 'test':
                transform = transforms.Compose([
                    Resize((320, 320)),
                    ToTensor1(),
                ])
        self.transform = transform
        if self.mode == 'train':
           class_frequencies = np.array([100.00, 18.33, 27.53, 9.26, 45.01, 89.82, 49.77])  # 假设每个类别的频率都是1，实际中需要计算
           repeat_factors = compute_repeat_factors(class_frequencies)

           self.frame_weights = compute_frame_weights(self.labeled_indices, repeat_factors)
    def __getitem__(self, index):
        # 返回带标注的数据
        sequence_path, frame, gt_path = self.labeled_indices[index]
        label = self.load_label(gt_path, frame)  # 加载标注
        if label is not None:
            label = Image.fromarray(label)
        img = self.load_image(sequence_path, frame)

        data = {'image': img, 'label': label}
        if self.mode == 'train':
            if self.transform:
                data = self.transform(data)
                to_tensor = transforms.ToTensor()

                img_s1 = Image.fromarray(np.array((data['image'])).astype(np.uint8))
                if random.random() < 0.8:
                    img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
                img_s1 = blur(img_s1, p=0.5)

                img_s1 = to_tensor(img_s1)
                label_image = data['label']

                label_array = np.array(label_image)
                label_tensor = torch.tensor(label_array, dtype=torch.long)

                if label_tensor.ndim == 2:  # 检查是否为二维数组
                    label_tensor = label_tensor.unsqueeze(0)  # 增加一个维度

                data['label'] = label_tensor
                data['image'] = to_tensor(data['image'])

                return {'image': data['image'], 'label': data['label'], 'image_s': img_s1}

        else:
            if self.transform:
                data = self.transform(data)
            return {'image': data['image'], 'label': data['label']}

    def load_image(self, sequence_path, frame):
        img_path = os.path.join(sequence_path, frame)
        return Image.open(img_path).convert('RGB')

    def load_label(self, gt_path, frame):
        label_path = os.path.join(gt_path, frame.replace('.jpg', '.png'))
        label_rgb = Image.open(label_path).convert('RGB')
        label_array = np.array(label_rgb)

        label_encoded = np.zeros(label_array.shape[:2], dtype=np.uint8)

        color_to_label = {
            (0, 0, 0): 0,  # 背景
            (128, 0, 0): 1,  # relic
            (0, 128, 0): 2,  # urchin
            (128, 128, 0): 3,  # starfish
            (0, 0, 128): 4,  # fish
            (128, 0, 128): 5,  # reefs
            (0, 128, 128): 6,  # dead coral
        }

        for color, label in color_to_label.items():
            mask = np.all(label_array == color, axis=-1)
            label_encoded[mask] = label

        return label_encoded

    def set_seed(self, seed):
        """
        固定随机数种子，确保每次运行结果一致
        """
        random.seed(seed)  # Python random
        np.random.seed(seed)  # Numpy random

    def __len__(self):
        return len(self.labeled_indices)

    def get_labeled_length(self):
        return len(self.labeled_indices)
