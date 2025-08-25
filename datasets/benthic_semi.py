import os
import torch
from utils.transform import *
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
# KavSir-SEG Dataset
# KavSir-SEG Dataset
class Benthic_img(Dataset):
    def __init__(self, root, data2_dir, mode='train', transform=None, cache=False):
        super(Benthic_img, self).__init__()
        self.data_path = os.path.join(root, data2_dir)

        self.id_list = []
        self.img_list = []
        self.gt_list = []
        self.mode = mode

        self.images_list = os.listdir(os.path.join(self.data_path, 'images'))  # images folder
        self.images_list = sorted(self.images_list)

        for img_id in self.images_list:
            self.id_list.append(img_id.split('.')[0])
            self.img_list.append(os.path.join(self.data_path, 'images', img_id))  # Image paths
            self.gt_list.append(os.path.join(self.data_path, 'masks', img_id))  # Mask paths



        if transform is None:
            if mode == 'train':
                transform = transforms.Compose([
                    Resize((640, 360)),
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                    RandomRotation(90),
                    RandomZoom((0.9, 1.1)),
                ])
            elif mode == 'valid' or mode == 'test':
                transform = transforms.Compose([
                    Resize((640, 360)),
                    ToTensor(),
                ])
        self.transform = transform

        self.cache = cache
        if self.cache and mode == 'train':
            self.cache_img = list()
            self.cache_gt = list()
            for index in range(len(self.img_list)):
                img_path = self.img_list[index]
                gt_path = self.gt_list[index]

                self.cache_img.append(Image.open(img_path).convert('RGB'))
                self.cache_gt.append(Image.open(gt_path).convert('L'))
        elif self.cache and (mode == 'valid' or mode == 'test'):
            self.cache_img = list()
            self.cache_gt = list()
            for index in range(len(self.img_list)):
                img_path = self.img_list[index]
                gt_path = self.gt_list[index]

                self.cache_img.append(Image.open(img_path).convert('RGB'))
                self.cache_gt.append(Image.open(gt_path).convert('L'))

    def __getitem__(self, index):
        if self.cache:
            img = self.cache_img[index]
            gt = self.cache_gt[index]
        else:
            img_path = self.img_list[index]
            gt_path = self.gt_list[index]

            img = Image.open(img_path).convert('RGB')
            gt = Image.open(gt_path).convert('L')


        data = {'image': img, 'label': gt}
        if self.mode == 'train':
            data = {'image': img, 'label': gt} # Add depth data in train mode

            if self.transform:
                data = self.transform(data)
                to_tensor = transforms.ToTensor()
                data['label'] = to_tensor(data['label'])
                img_s1 = Image.fromarray(np.array((data['image'])).astype(np.uint8))
                if random.random() < 0.8:
                    img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
                img_s1 = blur(img_s1, p=0.5)
                #cutmix_box1 = obtain_cutmix_box(320, p=0.5)

                img_s1 = to_tensor(img_s1)
                data['image'] = to_tensor(data['image'])

                return {'id': self.id_list[index], 'image': data['image'], 'label': data['label'], 'image_s': img_s1}
        else:
            if self.transform:
                data = self.transform(data)
            return {'id': self.id_list[index], 'image': data['image'], 'label': data['label']}

    def __len__(self):
        return len(self.img_list)
