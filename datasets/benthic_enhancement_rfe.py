import os
import torch
from models.utils.transform import *
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
from PIL import ImageFilter
from PIL import Image
import cv2
import numpy as np
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
def compute_gradients(image):
    """
    计算图像在 x 和 y 方向上的梯度。
    """
    # 使用 Sobel 算子计算梯度
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # x 方向梯度
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # y 方向梯度
    return grad_x, grad_y

def process_image(image_path):
    """
    处理图像并返回灰度图像以及其在 x 和 y 方向上的梯度。
    """
    # 读取图像
    image = cv2.imread(image_path)

    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算图像的 x 和 y 方向的梯度
    grad_x, grad_y = compute_gradients(gray_image)

    gray_image_pil = Image.fromarray(gray_image)
    grad_x_pil = Image.fromarray(grad_x.astype(np.uint8))  # 确保是uint8类型
    grad_y_pil = Image.fromarray(grad_y.astype(np.uint8))  # 确保是uint8类型

    return gray_image_pil, grad_x_pil, grad_y_pil

excluded_videos = [
    "video1", "video13", "video101", "video118", "video119", "video124",
    "video129", "video132", "video137", "video14", "video25", "video38",
    "video42", "video43", "video51", "video66", "video84", "video86", "video92", "video98"
]
# KavSir-SEG Dataset
# KavSir-SEG Dataset
class SynchronizedTransform1:
    """确保对图像和标签应用相同变换的类。"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data):
        img, label = data['image'],data['label']
        seed = torch.seed()  # 确保相同随机种子
        torch.manual_seed(seed)
        img = self.transform(img)
        torch.manual_seed(seed)
        label = self.transform(label)
        return img, label


class SynchronizedTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data):
        """
        data: 字典 {'image': img, 'gray': gray_image, 'x': x_image, 'y': y_image, 'label': label}
        对图像和标签中的所有项应用相同的变换
        """
        # 对字典中的每个元素应用相同的变换
        for key in data:
            data[key] = self.transform(data[key])

        return data
class BenthicVideoDataset_Enh(Dataset):
    def __init__(self, root, data2_dir, mode='train', transform=None):
        super(BenthicVideoDataset_Enh, self).__init__()
        self.root = root
        self.mode = mode
        self.data_path = os.path.join(root, data2_dir)
        self.video_dirs = sorted(os.listdir(self.data_path))  # 读取视频文件夹
        self.labeled_indices = []
        self.seed = 42
        self.set_seed(self.seed)

        #if self.mode == 'train':
            # 如果是 train 模式，只加载排除的视频
        #   self.video_dirs = [video for video in self.video_dirs if video in excluded_videos]
        # 遍历每个视频
        for video in self.video_dirs:

            video_path = os.path.join(self.data_path, video)
            gt_path = os.path.join(video_path, 'sequence_enh')  # 标注路径
            sequence_path = os.path.join(video_path, 'sequence')  # 图像路径

            frames = sorted(os.listdir(sequence_path))
            gt_frames = sorted(os.listdir(gt_path))  # 获取标注帧集合，转换为 JPEG 名称

            # 遍历所有帧，生成带标注的样本
            for frame in frames:
                if frame in gt_frames:  # 检查当前帧是否有对应的增强图像标签
                    self.labeled_indices.append((sequence_path, frame, gt_path))  # 添加到带标注的列表
        # 随机打乱 labeled_indices
        random.shuffle(self.labeled_indices)

        if transform is None:
            if mode == 'train':
                # 对输入和标签同时应用相同的变换
                transform = SynchronizedTransform(transforms.Compose([
                    transforms.Resize((320, 320)),  # 调整尺寸
                    #transforms.RandomHorizontalFlip(),
                    #transforms.RandomVerticalFlip(),
                    #transforms.RandomRotation(90),
                    transforms.ToTensor(),
                ]))
            elif mode in ['valid', 'test']:
                # 验证和测试模式只应用基本变换
                transform = SynchronizedTransform(transforms.Compose([
                    transforms.Resize((320, 320)),
                    transforms.ToTensor(),
                ]))
        self.transform = transform

    def __getitem__(self, index):
        # 返回带标注的数据
        sequence_path, frame, gt_path = self.labeled_indices[index]
        label = self.load_label(gt_path, frame)  # 加载标注

        img = self.load_image(sequence_path, frame)
        gray_image, grad_x, grad_y = process_image(os.path.join(sequence_path, frame))
        data = {'image': img, 'gray': gray_image, 'x':grad_x,'y':grad_y,'label': label}
        if self.mode == 'train':
            if self.transform:
                transformed_data = self.transform(data)

            return transformed_data

        else:
            if self.transform:
               date = self.transform(data)
            return date

    def load_image(self, sequence_path, frame):
        img_path = os.path.join(sequence_path, frame)
        return Image.open(img_path).convert('RGB')

    def load_label(self, gt_path, frame):
        img_path = os.path.join(gt_path, frame)
        return Image.open(img_path).convert('RGB')

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
