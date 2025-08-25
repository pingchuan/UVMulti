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
def compute_category_weights(category_counts):
    """
    根据每个类别的样本数，计算类别权重（样本数越少，权重越大）。
    """
    total_count = sum(category_counts.values())
    weights = {cat: total_count / count for cat, count in category_counts.items()}
    # 归一化权重
    weight_sum = sum(weights.values())
    return {cat: weight / weight_sum for cat, weight in weights.items()}


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
class BenthicVideoDataset_SUP_CP(Dataset):
    def __init__(self, root, data2_dir, mode='train', transform=None):
        super(BenthicVideoDataset_SUP_CP, self).__init__()
        self.root = root
        self.mode = mode
        self.data_path = os.path.join(root, data2_dir)
        self.video_dirs = sorted(os.listdir(self.data_path))  # 读取视频文件夹
        self.labeled_indices = []
        self.seed = 100
        self.set_seed(self.seed)
        self.Q1 = []  # 包含鱼（fish）的样本
        self.Q2 = []  # 包含海胆（urchin）的样本
        self.Q3 = [] #dead coral
        #if self.mode == 'train':
            # 如果是 train 模式，只加载排除的视频
         #   self.video_dirs = [video for video in self.video_dirs if video in excluded_videos]
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

        target_colors = {
            "fish": (0, 0, 128),  # fish 的 RGB 颜色
            "urchin": (0, 128, 0), # urchin 的 RGB 颜色
            "dead coral":(0, 128, 128)
        }
        if self.mode == 'train':
          for sequence_path, frame, gt_path in self.labeled_indices:
            gt_file = os.path.join(gt_path, frame.replace('.jpg', '.png'))  # 对应的标注文件
            gt_mask = np.array(Image.open(gt_file).convert('RGB'))  # 读取标注文件为 RGB 格式

            # 判断是否存在目标 RGB 颜色
            if self.contains_color(gt_mask, target_colors["fish"]):  # 是否存在鱼
                self.Q1.append((sequence_path, frame, gt_path))
            if self.contains_color(gt_mask, target_colors["urchin"]):  # 是否存在海胆
                self.Q2.append((sequence_path, frame, gt_path))
            if self.contains_color(gt_mask, target_colors["dead coral"]):  # 是否存在鱼
                self.Q3.append((sequence_path, frame, gt_path))




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
                if random.random() < 0.5:
                    img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
                if random.random() < 0.5:
                   img_s1 = blur(img_s1, p=0.5)

                label_image = data['label']
                if random.random() < 0.8:
                    img_s1,label_image = self.apply_cutmix1(img_s1, label_image, self.Q1, target_categories=[3])
                if random.random() < 0.6:
                        img_s1, label_image = self.apply_cutmix1(img_s1, label_image, self.Q3, target_categories=[5])
                if random.random() < 0.4:
                    img_s1, label_image = self.apply_cutmix1(img_s1, label_image, self.Q2, target_categories=[2])
                img_s1 = to_tensor(img_s1)

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
            (128, 128, 0): 0,  # starfish
            (0, 0, 128): 3,  # fish
            (128, 0, 128): 4,  # reefs
            (0, 128, 128): 5,  # dead coral
        }

        for color, label in color_to_label.items():
            mask = np.all(label_array == color, axis=-1)
            label_encoded[mask] = label

        return label_encoded


    def apply_cutmix1(self, img_s1, label_image, Q1, target_categories, img_size=320):
        """
        对输入图像和标签应用 CutMix，从 Q1 中随机选择一个样本作为前景，并确保前景区域包含目标类别。
        输入:
            img_s1: 输入图像 (PIL Image)
            label_image: 输入标签图像 (灰度图，NumPy 数组)
            Q1: 包含前景样本的列表，每项为 (sequence_path, frame, gt_path)
            target_categories: 目标类别列表，确保前景区域包含这些类别
            img_size: 图像和标签的目标尺寸
        输出:
            处理后的 img_s1 和 label_image
        """

        # 随机从 Q1 中选择一个前景样本
        while True:
            fg_sequence_path, fg_frame, fg_gt_path = random.choice(Q1)

            # 加载前景图像和标签
            fg_image = self.load_image(fg_sequence_path, fg_frame)
            fg_label = self.load_label(fg_gt_path, fg_frame)

            # 检查前景掩码是否包含目标类别
            if np.any(np.isin(fg_label, target_categories)):  # 确保前景中包含目标类别
                break
        from scipy import ndimage

        # 调整前景图像和标签的大小
        fg_image_resized = fg_image.resize((img_size, img_size), Image.BILINEAR)
        fg_label_resized = np.array(Image.fromarray(fg_label).resize((img_size, img_size), Image.NEAREST))

        # 调整背景图像和标签的大小
        img_s1_resized = img_s1.resize((img_size, img_size), Image.BILINEAR)
        #label_image_resized = np.array(Image.fromarray(label_image).resize((img_size, img_size), Image.NEAREST))
        label_image_resized_pil = label_image.resize((img_size, img_size), Image.NEAREST)

        # 将调整大小后的 PIL 图像转换回 NumPy 数组
        label_image_resized = np.array(label_image_resized_pil)
        # 生成前景掩码，确保目标类别在前景区域
        fg_mask = np.isin(fg_label_resized, target_categories).astype(np.uint8)

        # 对前景掩码进行标记，获取连通区域
        labeled_array, num_features = ndimage.label(fg_mask)  # 对掩码进行标记，获取连通区域和区域数量

        if num_features == 0:
            return img_s1, label_image  # 如果没有目标类别区域，直接返回原图

        # 随机选择一个区域
        random_region = random.choice(range(1, num_features + 1))  # 从1到num_features选择一个区域

        # 获取该区域的坐标
        region_indices = np.where(labeled_array == random_region)

        # 获取区域的边界
        top = np.min(region_indices[0])
        left = np.min(region_indices[1])
        bottom = np.max(region_indices[0])
        right = np.max(region_indices[1])

        # 确保区域在图像内
        height, width = fg_image_resized.size
        bottom = min(bottom, height)
        right = min(right, width)

        # 创建一个包含区域的cutmix掩码
        cutmix_mask = np.zeros((img_size, img_size), dtype=np.uint8)
        cutmix_mask[top:bottom, left:right] = 1  # 在选择的区域内设置为1

        # 扩展CutMix掩码的维度以适应图像通道
        cutmix_mask1 = np.expand_dims(cutmix_mask, axis=-1)  # [H, W, 1]

        # 使用CutMix掩码进行混合
        combined_image = cutmix_mask1 * np.array(fg_image_resized) + (1 - cutmix_mask1) * np.array(img_s1_resized)
        combined_label = cutmix_mask * fg_label_resized + (1 - cutmix_mask) * label_image_resized

        # 转换回PIL格式
        combined_image = Image.fromarray(combined_image.astype(np.uint8))
        combined_label = Image.fromarray(combined_label.astype(np.uint8))

        return combined_image, combined_label


    def apply_cutmix(self, img_s1, label_image, Q1, target_categories, img_size=320):
        """
        对输入图像和标签应用 CutMix，从 Q1 中随机选择一个样本作为前景，并确保前景区域包含目标类别。
        输入:
            img_s1: 输入图像 (PIL Image)
            label_image: 输入标签图像 (灰度图，NumPy 数组)
            Q1: 包含前景样本的列表，每项为 (sequence_path, frame, gt_path)
            target_categories: 目标类别列表，确保前景区域包含这些类别
            img_size: 图像和标签的目标尺寸
        输出:
            处理后的 img_s1 和 label_image
        """

        # 随机从 Q1 中选择一个前景样本
        while True:
            fg_sequence_path, fg_frame, fg_gt_path = random.choice(Q1)

            # 加载前景图像和标签
            fg_image = self.load_image(fg_sequence_path, fg_frame)
            fg_label = self.load_label(fg_gt_path, fg_frame)

            # 检查前景掩码是否包含目标类别
            if np.any(np.isin(fg_label, target_categories)):  # 确保前景中包含目标类别
                break

        # 调整前景图像和标签的大小
        fg_image_resized = fg_image.resize((img_size, img_size), Image.BILINEAR)
        fg_label_resized = np.array(Image.fromarray(fg_label).resize((img_size, img_size), Image.NEAREST))

        # 调整背景图像和标签的大小
        img_s1_resized = img_s1.resize((img_size, img_size), Image.BILINEAR)
        #label_image_resized = np.array(Image.fromarray(label_image).resize((img_size, img_size), Image.NEAREST))
        label_image_resized_pil = label_image.resize((img_size, img_size), Image.NEAREST)

        # 将调整大小后的 PIL 图像转换回 NumPy 数组
        label_image_resized = np.array(label_image_resized_pil)
        # 生成前景掩码，确保目标类别在前景区域
        fg_mask = np.isin(fg_label_resized, target_categories).astype(np.uint8)

        # 生成一个覆盖前景区域的切割区域
        fg_indices = np.where(fg_mask == 1)  # 获取前景区域的索引
        if len(fg_indices[0]) == 0:  # 如果没有目标类别的前景区域，跳过
            return img_s1, label_image

        # 获取前景区域的极端顶部和最左侧
        top = np.min(fg_indices[0])  # 获取最上面的行索引
        left = np.min(fg_indices[1])  # 获取最左边的列索引

        height = random.randint(10, 50)  # 随机选择切割区域的高度
        width = random.randint(10, 50)  # 随机选择切割区域的宽度

        # 确保选定的区域在图像内
        bottom = min(top + height, fg_image_resized.size[1])
        right = min(left + width, fg_image_resized.size[0])

        # 创建 CutMix 区域掩码
        cutmix_mask = np.zeros((img_size, img_size), dtype=np.uint8)
        cutmix_mask[top:bottom, left:right] = 1  # 在选择的区域内设置为 1

        # 扩展 CutMix 掩码的维度以适应图像通道
        cutmix_mask1 = np.expand_dims(cutmix_mask, axis=-1)  # [H, W, 1]

        # 使用 CutMix 掩码进行混合
        combined_image = cutmix_mask1 * np.array(fg_image_resized) + (1 - cutmix_mask1) * np.array(img_s1_resized)
        combined_label = cutmix_mask * fg_label_resized + (1 - cutmix_mask) * label_image_resized

        # 转换回 PIL 格式
        combined_image = Image.fromarray(combined_image.astype(np.uint8))
        combined_label = Image.fromarray(combined_label.astype(np.uint8))

        return combined_image, combined_label

    def contains_color(self, mask, color):
        """
        判断 RGB 掩码中是否存在特定颜色像素。
        """
        return np.any(np.all(mask == color, axis=-1))

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
