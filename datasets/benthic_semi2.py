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

# KavSir-SEG Dataset
# KavSir-SEG Dataset
class BenthicVideoDataset2(Dataset):
    def __init__(self, root, data2_dir, mode='train', transform=None, only_labeled=False):
        super(BenthicVideoDataset2, self).__init__()
        self.root = root
        self.mode = mode
        self.data_path = os.path.join(root, data2_dir)
        self.video_dirs = sorted(os.listdir(self.data_path))  # 读取视频文件夹
        self.labeled_indices = []
        self.unlabeled_indices = []
        self.seed = 8888
        self.set_seed(self.seed)




        # 遍历每个视频
        for video in self.video_dirs:
            video_path = os.path.join(self.data_path, video)
            gt_path = os.path.join(video_path, 'mask')  # 标注路径
            sequence_path = os.path.join(video_path, 'sequence')  # 图像路径

            frames = sorted(os.listdir(sequence_path))
            gt_frames = {f.replace('.png', '.jpg') for f in os.listdir(gt_path)}  # 获取标注帧集合，转换为 JPEG 名称

            # 只保留首帧、每15帧和最后一帧作为带标注的帧
            total_frames = len(frames)
            if total_frames > 0:
                # 保留首帧
                if frames[0] in gt_frames:  # 检查首帧
                    self.labeled_indices.append((sequence_path, frames[0], gt_path))

                # 保留每15帧
                for i in range(1, total_frames - 1):
                    if i % 15 == 0 and frames[i] in gt_frames:  # 检查每15帧
                        self.labeled_indices.append((sequence_path, frames[i], gt_path))

                # 保留最后一帧
                if frames[-1] in gt_frames:  # 检查最后一帧
                    self.labeled_indices.append((sequence_path, frames[-1], gt_path))

            # 其他帧都是无标注的
            for frame in frames:
                if frame not in gt_frames:
                    self.unlabeled_indices.append((sequence_path, frame))  # 添加到不带标注的帧中

        # 随机打乱 labeled_indices 和 unlabeled_indices
        random.shuffle(self.labeled_indices)
        random.shuffle(self.unlabeled_indices)

        if only_labeled and mode == 'valid':
            self.unlabeled_indices = []

        if transform is None:
            if mode == 'train':
                transform = transforms.Compose([
                    Resize2((320, 320)), #1280,720 四次降采样正好
                    RandomHorizontalFlip2(),
                    RandomVerticalFlip2(),
                    RandomRotation2(90),
                    #RandomZoom((0.9, 1.1)),
                ])
            elif mode == 'valid' or mode == 'test':
                transform = transforms.Compose([
                    Resize((320, 320)),
                    ToTensor1(),
                ])
        self.transform = transform

    def __getitem__(self, index):
        # 确定返回带标注还是不带标注的数据
        if index < len(self.labeled_indices):
            sequence_path, frame, gt_path = self.labeled_indices[index]
            label = self.load_label(gt_path, frame)  # 加载标注
            if label is not None:
                # 将标签转换为张量并添加一个维度
                label = np.array(label).astype(np.uint8) # 转换为 numpy 数组
                #label = torch.tensor(label, dtype=torch.long)  # 转换为长整型张量
                #label = torch.from_numpy(label).unsqueeze(0)
                #label_tensor = label_tensor.unsqueeze(0)  # 添加一个通道维度，变成 (1, H, W)
                #label = label_tensor
                label = Image.fromarray(label)
            img = self.load_image(sequence_path, frame)
        else:
            idx = index - len(self.labeled_indices)
            sequence_path, frame = self.unlabeled_indices[idx]
            label = np.zeros((320, 320), dtype=np.uint8)
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
              if random.random() < 0.8:
                 img_s1 = blur(img_s1, p=0.5)
              # cutmix_box1 = obtain_cutmix_box(320, p=0.5)

              img_s1 = to_tensor(img_s1)
              label_image = data['label']

              # 将 PIL 图像转换为 NumPy 数组
              label_array = np.array(label_image)

              # 转换为长整型张量
              label_tensor = torch.tensor(label_array, dtype=torch.long)

              # 如果是灰度图像，确保输出形状为 [1, H, W]
              if label_tensor.ndim == 2:  # 检查是否为二维数组
                  label_tensor = label_tensor.unsqueeze(0)  # 增加一个维度

              # 如果需要确保是整数类型（如 uint8），可以使用以下方式
              data['label'] = label_tensor
              data['image'] = to_tensor(data['image'])

              #print(data['label'].max())
              return {'image': data['image'], 'label': data['label'], 'image_s': img_s1}
        # 在训练模式下生成图像增强版
        else:
            if self.transform:
                data = self.transform(data)
            return {'image': data['image'], 'label': data['label']}

    def load_image(self, sequence_path, frame):
        img_path = os.path.join(sequence_path, frame)
        return Image.open(img_path).convert('RGB')

    def load_label(self, gt_path, frame):
        label_path = os.path.join(gt_path, frame.replace('.jpg', '.png'))
        #label_path = os.path.join(gt_path, frame)  # 假设标注在 gt 文件夹中
        label_rgb = Image.open(label_path).convert('RGB')
        label_array = np.array(label_rgb)

        # 创建一个与标签图像大小相同的单通道图像
        label_encoded = np.zeros(label_array.shape[:2], dtype=np.uint8)

        # 创建一个映射字典
        color_to_label = {
            (0, 0, 0): 0,  # 背景
            (128, 0, 0): 1,  # relic
            (0, 128, 0): 2,  # urchin
            (128, 128, 0): 0,  # starfish
            (0, 0, 128): 3,  # fish
            (128, 0, 128): 4,  # reefs
            (0, 128, 128): 5,  # dead coral
        }

        # 使用向量化操作将 RGB 映射到单通道编码
        for color, label in color_to_label.items():
            mask = np.all(label_array == color, axis=-1)
            label_encoded[mask] = label
            # 处理未映射的颜色

        return Image.fromarray(label_encoded, mode='L')

    def set_seed(self, seed):
        """
        固定随机数种子，确保每次运行结果一致
        """
        random.seed(seed)  # Python random
        np.random.seed(seed)  # Numpy random

    def __len__(self):
        # 返回带标注和不带标注的样本总数
        return len(self.labeled_indices) + len(self.unlabeled_indices)

    def get_labeled_length(self):
        return len(self.labeled_indices)

    def get_unlabeled_length(self):
        return len(self.unlabeled_indices)