from __future__ import absolute_import, division, print_function

import os
from PIL import Image  # using pillow-simd for increased speed
import skimage
import random
import numpy as np
from PIL import Image  # using pillow-simd for increased speed
import skimage
import torch
import torch.utils.data as data
from torchvision import transforms
import torch.nn as nn
from datasets.ResNet34UNet1 import ResNet34U1
import tifffile

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

from torch.utils.data import Sampler
import numpy as np

class CustomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, n_label, shuffle=True):
        """
        Custom batch sampler to ensure each batch contains:
        - n_label labeled samples
        - batch_size - n_label unlabeled samples

        Args:
            dataset: The dataset to sample from.
            batch_size: The total batch size.
            n_label: The number of labeled samples per batch.
            shuffle: Whether to shuffle the data or not.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_label = n_label
        self.shuffle = shuffle

        # 创建两个列表，一个用于标签数据，一个用于无标签数据
        self.labeled_indices = list(range(len(self.dataset.filenames_with_labels)))
        self.unlabeled_indices = list(range(len(self.dataset.filenames_without_labels)))

        # 如果需要 shuffle 数据，打乱索引顺序
        if self.shuffle:
            np.random.shuffle(self.labeled_indices)
            np.random.shuffle(self.unlabeled_indices)

    def __iter__(self):
        # 创建一个批次的生成器
        for i in range(0, len(self.labeled_indices), self.batch_size):
            labeled_batch = self.labeled_indices[i:i+self.n_label]  # 前 n_label 个标签数据
            unlabeled_batch = self.unlabeled_indices[i:i+(self.batch_size - self.n_label)]  # 后 batch_size - n_label 个无标签数据

            # 合并标签数据和无标签数据
            batch_indices = labeled_batch + unlabeled_batch

            # 如果标签数据不足，填充无标签数据
            if len(labeled_batch) < self.n_label:
                remaining = self.n_label - len(labeled_batch)
                batch_indices = labeled_batch + self.unlabeled_indices[:remaining] + unlabeled_batch

            # 返回生成的批次索引
            yield batch_indices

    def __len__(self):
        # 返回批次的总数
        return (len(self.labeled_indices) + len(self.unlabeled_indices)) // self.batch_size

import random
class BenthicDataset_VS(data.Dataset):
    """Benthic Dataset
    Args:
        data_path
        filenames
        height
        width
        num_scales
        is_train
        img_ext
        load_depth
        load_enhanced_img
    """

    def __init__(self,
                 data_path,
                 filenames_with_labels,
                 filenames_without_labels,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 img_ext='.jpg',
                 is_train=False,
                 load_depth=False,
                 load_enh_img=False,
                 load_gt=False,
                 data2_dir='train'):
        super(BenthicDataset_VS, self).__init__()

        self.filenames_with_labels = filenames_with_labels  # 有标签的文件
        self.filenames_without_labels = filenames_without_labels  # 无标签的文件

        self.data_path = data_path
        self.data2_dir = data2_dir
        self.data_path = os.path.join(data_path, data2_dir)
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.is_train = is_train
        self.img_ext = img_ext
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.load_depth = load_depth
        self.load_enh_img = load_enh_img
        self.load_gt = load_gt
        self.frame_idxs = frame_idxs
        self.K = np.array([[0.70859375, 0, 0.5, 0],
                           [0, 1.25972222, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.full_res_shape = (1280, 720)

        # Augmentation params
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)

        # Resize transformations
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.resize[1] = transforms.Resize((self.height, self.width),
                                       interpolation=transforms.InterpolationMode.NEAREST)
    def preprocess(self, inputs, color_aug):
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
            if "enh" in k:
                n = k
                inputs[n] = self.resize[0](inputs[n])
            if "gt" in k:
                n = k
                inputs[n] = self.resize[1](inputs[n])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
            if "enh" in k:
                n = k
                inputs[n] = self.to_tensor(f)
            if "gt" in k:
                n = k
                image_array = np.array(inputs[n], dtype=np.uint8)
                inputs[n] = torch.from_numpy(image_array).long()

    def del_useless(self, inputs, load_enh, load_gt):
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            # del inputs[("color_aug", i, -1)]

    def __len__(self):
        # 返回有标签和无标签图像的总数
        return len(self.filenames_with_labels) + len(self.filenames_without_labels)

    def __getitem__(self, index):
        inputs = {}

        # 确定当前数据是有标签还是无标签
        if index < len(self.filenames_with_labels):
            # 有标签数据
            filenames = self.filenames_with_labels
            has_label = True
        else:
            # 无标签数据
            filenames = self.filenames_without_labels
            has_label = False
            index -= len(self.filenames_with_labels)  # 调整索引以便访问无标签文件
        do_color_aug = self.is_train and random.random() > 0.5
        # 获取图像信息
        line = filenames[index].split()
        folder = line[0]
        frame_index = line[1]
        frame_index_int = int(line[1])
        img_folder = 'imgs'

        inputs[("color", 0, -1)] = self.get_image(folder, frame_index, img_folder)

        # 无标签图像，填充全0标签
        if not has_label:
            label_encoded = np.zeros((self.height, self.width), dtype=np.uint8)
            inputs["gt"] = Image.fromarray(label_encoded, mode='L')
        else:
            # 有标签图像，加载真实标签
            inputs["gt"] = self.get_gt_t(folder, frame_index, img_folder)

        if True:
            if len(line) == 3:
                if line[2] == 'start' or line[2] == 'end':
                    if line[2] == 'start':
                        # 如果是 "start"，选择下一个帧
                        next_frame_index = frame_index_int + 1
                        frame_index_after = f"{next_frame_index:05d}"  # 格式化为五位数，确保数字始终为五位
                        frame_index_before = f"{next_frame_index:05d}"
                        frame_index_before2 = f"{next_frame_index:05d}"

                    else:
                        # 如果是 "end"，选择前一个帧
                        previous_frame_index = frame_index_int - 1
                        frame_index_before = f"{previous_frame_index:05d}"
                        frame_index_after = f"{previous_frame_index:05d}"
                        frame_index_before2 = f"{previous_frame_index:05d}"

            else:
                next_frame_index = frame_index_int + 1
                previous_frame_index = frame_index_int - 1
                previous_frame_index2 = frame_index_int - 2
                frame_index_before = f"{previous_frame_index:05d}"
                frame_index_after = f"{next_frame_index:05d}"
                frame_index_before2 = f"{previous_frame_index2:05d}"

            inputs[("color", 1, -1)] = self.get_image(folder, frame_index_after, img_folder)
            inputs[("color", -1, -1)] = self.get_image(folder, frame_index_before, img_folder)
            inputs[("color", -2, -1)] = self.get_image(folder, frame_index_before2, img_folder)

        if self.load_enh_img:
            img_folder = 'IEB'
            inputs["enh"] = self.get_image_enh(folder, frame_index, img_folder)
        if self.load_gt:
            img_folder = 'IEB'
            inputs["gt"] = self.get_gt_t(folder, frame_index, img_folder)

        # camera intrinsics
        for scale in range(self.num_scales):
            K = self.K.copy()
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)
            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        # Augmentation
        do_color_aug = self.is_train and random.random() > 0.5
        color_aug = (lambda x: x) if not do_color_aug else (lambda x: x)
        self.preprocess(inputs, color_aug)

        # Load depth (if needed)
        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs


    def get_image_enh(self, folder, frame_index, image_folder):
        f_str = "{}{}".format(frame_index, self.img_ext)

        image_path = os.path.join(self.data_path, folder.replace('sequence', 'sequence_enh'), f_str)
        color = self.loader(image_path)

        return color
    def get_image(self, folder, frame_index, image_folder):
        f_str = "{}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f_str)
        color = self.loader(image_path)
        return color

    def get_gt_t(self, folder, frame_index, image_folder):
        gt_ext = '.png'
        color_to_label = {
            (0, 0, 0): 0,  # 背景
            (128, 0, 0): 1,  # relic
            (0, 128, 0): 2,  # urchin
            (128, 128, 0): 0,  # starfish
            (0, 0, 128): 3,  # fish
            (128, 0, 128): 4,  # reefs
            (0, 128, 128): 5,  # dead coral
        }
        f_str = "{}{}".format(frame_index, gt_ext)
        image_path = os.path.join(self.data_path, folder.replace('sequence', 'mask'), f_str)
        color = self.loader(image_path)
        label_array = np.array(color)
        label_encoded = np.zeros(label_array.shape[:2], dtype=np.uint8)
        for color, label in color_to_label.items():
            mask = np.all(label_array == color, axis=-1)
            label_encoded[mask] = label
        return Image.fromarray(label_encoded, mode='L')

    def get_depth(self, folder, frame_index):
        folder1 = folder.replace('sequence', '')
        velo_filename = os.path.join(self.data_path, folder1, "depth/{}.tif".format(frame_index))
        depth_gt = tifffile.imread(velo_filename)
        if depth_gt.dtype != np.float32:
            depth_gt = depth_gt.astype(np.float32)
        depth_gt = skimage.transform.resize(depth_gt, self.full_res_shape[::-1], order=1, mode='reflect', cval=0.0,
                                            clip=True)
        return depth_gt