# Based on https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/cityscapes_loader.py

import os

import numpy as np

from loader.sequence_segmentation_loader1 import SequenceSegmentationLoader
from utils.utils import recursive_glob,recursive_glob_find


class Benthic:
    n_classes = 5  # 只考虑有效类别，不包括背景
    #ignore_index = -1  # 定义忽略的索引，假设为 -1

    colors = [
        [0,   0,   0],    # 背景
        [128, 0,   0],    # relic
        [0,   128, 128],    # urchin ###不确定
        [128, 128,   0],    # starfish
        [0, 0,   128],    # fish
        [128,   0,   128],    # reefs
    ]

    label_colours = dict(zip(range(n_classes + 1), colors))  # +1 包含背景颜色

    # void_classes 可以设为空列表，因为没有需要忽略的类别
    void_classes = []
    valid_classes = [0, 1, 2, 3, 4, 5]  # 有效类别的索引

    class_names = [
        "__background__",  # 0
        "relic",          # 1
        "urchin",        # 2
        "starfish",      # 3
        "fish",          # 4
        "reefs",         # 5
    ]

    class_map = {i + 1: i for i in range(n_classes)}  # 1 to n_classes map
    decode_class_map = {i: i + 1 for i in range(n_classes)}  # 0 to n_classes

    @staticmethod
    def decode_segmap_tocolor(temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(Benthic.n_classes + 1):  # +1 包括背景
            r[temp == l] = Benthic.label_colours[l][0]
            g[temp == l] = Benthic.label_colours[l][1]
            b[temp == l] = Benthic.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    @staticmethod
    def encode_segmap(mask):
        # 由于没有需要忽略的类别，这里可以直接返回
        for _validc in Benthic.valid_classes:
            mask[mask == _validc] = Benthic.class_map[_validc]
        return mask


class BenthicLoader(SequenceSegmentationLoader):
    def __init__(self, **kwargs):
        super(BenthicLoader, self).__init__(**kwargs)

        self.n_classes = Benthic.n_classes
        #self.ignore_index = Benthic.ignore_index
        self.void_classes = Benthic.void_classes
        self.valid_classes = Benthic.valid_classes
        self.label_colors = Benthic.label_colours
        self.class_names = Benthic.class_names
        self.class_map = Benthic.class_map
        self.decode_class_map = Benthic.decode_class_map

        self.full_res_shape = (1280, 720)
        # See https://www.cityscapes-dataset.com/file-handling/?packageID=8
        self.fx = 2262.52
        self.fy = 2265.3017905988554
        self.u0 = 1096.98
        self.v0 = 513.137

    def _prepare_filenames(self):
        if self.img_size == (720, 1280):
            self.images_base = os.path.join(self.root, self.split)
            self.sequence_base = os.path.join(self.root, self.split)
        elif self.img_size == (360, 640):
            self.images_base = os.path.join(self.root, self.split)
            self.sequence_base = os.path.join(self.root, self.split)
        else:
            raise NotImplementedError(f"Unexpected image size {self.img_size}")
        self.annotations_base = os.path.join(self.root, self.split)

        if self.only_sequences_with_segmentation:
            self.files = sorted(recursive_glob_find(rootdir=self.images_base, target_folder="images"))
        else:
            self.files = sorted(recursive_glob_find(rootdir=self.sequence_base, target_folder="sequence"))

    def get_image_path(self, index, offset=0):
        img_path = self.files[index]["name"].rstrip()
        #if offset != 0:
         #   img_path = img_path.replace(self.images_base, self.sequence_base)
          #  prefix, frame_number, suffix = img_path.rsplit('_', 2)
          #  frame_number = int(frame_number)
           # img_path = f"{prefix}_{frame_number + offset:06d}_{suffix}"
        return img_path

    def get_segmentation_path(self, index):
        img_path = self.files[index]["name"].rstrip()
        segmentation_path = img_path.replace('sequence', 'gt')
        return segmentation_path

    def decode_segmap_tocolor(self, temp):
        return Benthic.decode_segmap_tocolor(temp)

    def encode_segmap(self, mask):
        return Benthic.encode_segmap(mask)
