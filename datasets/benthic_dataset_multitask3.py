from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
from PIL import Image  # using pillow-simd for increased speed
import skimage
import torch
import torch.utils.data as data
from torchvision import transforms
import torch.nn as nn
from datasets.ResNet34UNet1 import ResNet34U1

def generate_pseudo_labels(img_path, confidence_threshold=0.95, device='cuda', target_size=(256, 256)):
    """
    根据教师网络生成伪标签
    :param model_t: 训练好的教师网络模型
    :param img_path: 输入图像路径
    :param confidence_threshold: 置信度阈值，低于该值的像素将被忽略
    :param device: 使用的设备，默认为 'cuda'
    :param target_size: 调整图像的目标大小，默认为 (256, 256)
    :return: 伪标签 (pseudo_labels) 和 置信度掩码 (confidence_mask)
    """

    model_t = ResNet34U1(num_classes=6)
    model_weights_path = r"E:/python project/multitasks/PolypMix-main/checkpoints/kvasir_SEG_normal_ablation_toatl_d_1/Multi-semi_ResNet34U1/exp8888_10/checkpoint_best_iou.pth"
    model_t.load_state_dict(torch.load(model_weights_path))
    model_t.to(device)  # 将模型移到指定的设备上
    model_t.eval()
    # 定义图像预处理方式
    transform = transforms.Compose([
        transforms.Resize(target_size),  # 对输入图像进行resize
        transforms.ToTensor(),  # 转换为 tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 进行标准化
    ])

    # 加载图像并进行预处理
    image = Image.open(img_path).convert('RGB')
    image_resized = transform(image)  # 调整大小并进行预处理
    original_size = image.size  # 保存原始图像大小 (宽, 高)
    image_tensor = image_resized.unsqueeze(0).to(device)  # 增加 batch 维度，并移动到设备

    # 将图像输入到教师网络中
    model_t.eval()  # 切换到评估模式
    with torch.no_grad():  # 不需要计算梯度
        outputs = model_t(image_tensor)  # 假设输出是形状为 [B, C, H, W]，B=1

    # 获取 softmax 输出，形状为 [B, C, H, W]
    softmax_outputs = nn.Softmax(dim=1)(outputs)  # [B, C, H, W]

    # 对每个像素获取最大类别及其置信度
    confidences, pseudo_labels = torch.max(softmax_outputs, dim=1)  # confidences: [B, H, W], pseudo_labels: [B, H, W]

    # 选择置信度高于阈值的像素（否则设为 0）
    confidence_mask = (confidences >= confidence_threshold).float()

    # 使用插值将预测结果恢复到原始图像的大小
    pseudo_labels_resized = nn.functional.interpolate(pseudo_labels.unsqueeze(1).float(), size=original_size, mode='nearest').squeeze(1).long()
    confidence_mask_resized = nn.functional.interpolate(confidence_mask.unsqueeze(1).float(), size=original_size, mode='nearest').squeeze(1)

    # 将输出从 GPU 移回 CPU，并转为 numpy 格式
    pseudo_labels_resized = pseudo_labels_resized.squeeze().cpu().numpy()  # 移除 batch 维度
    confidence_mask_resized = confidence_mask_resized.squeeze().cpu().numpy()  # 移除 batch 维度

    return pseudo_labels_resized, confidence_mask_resized


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')




class BenthicDataset(data.Dataset):
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
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 img_ext='.jpg',
                 is_train=False,
                 load_depth=False,
                 load_enh_img=True,
                 load_gt = False,
                 data2_dir='train',
                 ):
        super(BenthicDataset, self).__init__()
        self.filenames = filenames
        self.data_path = data_path
        self.data2_dir = data2_dir
        self.data_path = os.path.join(data_path, data2_dir)
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.LANCZOS
        self.is_train = is_train
        self.img_ext = img_ext
        self.depth_ext = '.png'
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.load_depth = load_depth
        self.load_enh_img = load_enh_img
        self.load_gt = load_gt
        #self.frame_idxs = [0, -1, 1] if self.is_train else [0]
        self.frame_idxs = frame_idxs
        # Define the intrinsic matrix K
        self.K = np.array([[1.214, 0, 0.48, 0],
                           [0, 1.931, 0.44, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1280, 720)

        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}

        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]

            if "color" in k:
                #'color' refer to the original image
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)]) #zuihou('color',0,-1)0代表这一帧，-1代表原始resize的size，
            if "color_enh" in k:
                n, im, _ = k
                inputs[(n, im, 0)] = self.resize[0](inputs[(n, im, -1)])  #加一个gt，要考虑gt是灰度不能直接用to_tensor

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
            if "color_enh" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
            if "gt" in k:
                n, im, i = k
                image_array = np.array(inputs[(n, im, i)], dtype=np.uint8)
                # 将 numpy 数组转换为 Tensor，并确保它保持原始值
                inputs[(n, im, i)] = torch.from_numpy(image_array).long()
            #inputs[(n, im, i)] = self.to_tensor(f)
            if "gt_mask" in k:
                n, im, i = k
                image_array = np.array(inputs[(n, im, i)], dtype=np.uint8)
                # 将 numpy 数组转换为 Tensor，并确保它保持原始值
                inputs[(n, im, i)] = torch.from_numpy(image_array).long()
                #inputs[(n, im, i)] = self.to_tensor(f)

        #for k in list(inputs):
         #   print(k)
    def del_useless(self, inputs, load_enh,load_gt):
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
            if load_gt:
                del inputs[("gt", i, -1)]
                del inputs[("gt_mask", i, -1)]

            if load_enh:
                del inputs[("color_enh", i, -1)]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("color_enh_aug", <frame_id>, <scale>)  for augmented enhanced colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]
        frame_index = line[1]

        #print(index,folder,frame_index)
        img_folder = 'imgs'
        inputs[("color", 0, -1)] = self.get_image(folder, frame_index, img_folder)
        #if self.is_train:
        if True:
            if len(line) == 4:
                if line[2] == 'start' or line[2] == 'end':
                    frame_index_before = (self.filenames[index + 1 if line[2] == 'start' else index - 1].split()[1])
                    frame_index_after = (self.filenames[index + 1 if line[2] == 'start' else index - 1].split()[1])


            else:
                frame_index_before = (self.filenames[index - 1].split()[1])
                frame_index_after = (self.filenames[index + 1].split()[1])

            inputs[("color", 1, -1)] = self.get_image(folder, frame_index_after, img_folder)
            inputs[("color", -1, -1)] = self.get_image(folder, frame_index_before, img_folder)

        if self.load_enh_img:
            img_folder = 'IEB'
            #inputs[("color_enh", 0, -1)] = self.get_image_enh(folder, frame_index, img_folder)
            #inputs[("color_enh", 1, -1)] = self.get_image_enh(folder, frame_index_after, img_folder)
            #inputs[("color_enh", -1, -1)] = self.get_image_enh(folder, frame_index_before, img_folder)
            inputs["color_enh"] = self.get_image_enh(folder, frame_index, img_folder)
        if self.load_gt:
            img_folder = 'IEB'
            #inputs[("gt", 0, -1)], inputs[("gt_mask", 0, -1)] = self.get_gt_t(folder, frame_index, img_folder)
            #inputs[("gt", 1, -1)], inputs[("gt_mask", 1, -1)] = self.get_gt_t(folder, frame_index_after, img_folder)
            #inputs[("gt", -1, -1)], inputs[("gt_mask", -1, -1)] = self.get_gt_t(folder, frame_index_before, img_folder)
            inputs["gt"], inputs["gt_mask"] = self.get_gt_t(folder, frame_index, img_folder)
        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        self.del_useless(inputs, self.load_enh_img, self.load_gt)


        # load gt_depth
        if self.load_depth:
            if self.check_depth():
                depth_gt = self.get_depth(folder, frame_index)
                inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
                inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs

    def get_image_enh(self, folder, frame_index, image_folder):
        f_str = "{}{}".format(frame_index, self.img_ext)

        image_path = os.path.join(self.data_path, folder.replace('sequence', 'sequence_enh'), f_str)
        color = self.loader(image_path)

        return color

    import os
    import numpy as np
    from PIL import Image

    def get_gt_t(self, folder, frame_index, image_folder):
        gt_ext = '.png'
        imh_ext = '.jpg'
        # 创建一个映射字典
        color_to_label = {
            (0, 0, 0): 0,  # 背景
            (128, 0, 0): 1,  # relic
            (0, 128, 128): 2,  # urchin
            (128, 128, 0): 3,  # starfish
            (0, 0, 128): 4,  # fish
            (128, 0, 128): 5,  # reefs
        }

        f_str = "{}{}".format(frame_index, gt_ext)
        f_str1="{}{}".format(frame_index, imh_ext)
        img_path = os.path.join(self.data_path, folder, f_str1)
        # 修改路径，检查是否存在GT图像
        image_path = os.path.join(self.data_path, folder.replace('sequence', 'gt'), f_str)

        if os.path.exists(image_path):
            # 路径存在，加载 GT 图像
            color = self.loader(image_path)
            label_array = np.array(color)

            # 创建一个与标签图像大小相同的单通道图像
            label_encoded = np.zeros(label_array.shape[:2], dtype=np.uint8)

            # 使用向量化操作将 RGB 映射到单通道编码
            for color, label in color_to_label.items():
                mask = np.all(label_array == color, axis=-1)
                label_encoded[mask] = label

            # 置信度掩码，全为 1，因为是 GT 图像
            confidence_mask = np.ones_like(label_encoded, dtype=np.uint8)

            return Image.fromarray(label_encoded, mode='L'), confidence_mask
        else:

            # 路径不存在，使用教师网络生成伪标签
            # 假设使用教师网络进行伪标签生成并得到相应的置信度图（你需要提供这部分代码或函数）
            pseudo_labels, confidence_mask = generate_pseudo_labels(img_path, confidence_threshold=0.95, device='cuda')

            return pseudo_labels, confidence_mask



    def get_gt(self, folder, frame_index, image_folder):
        gt_ext = '.png'
        # 创建一个映射字典
        color_to_label = {
            (0, 0, 0): 0,  # 背景
            (128, 0, 0): 1,  # relic
            (0, 128, 128): 2,  # urchin
            (128, 128, 0): 3,  # starfish
            (0, 0, 128): 4,  # fish
            (128, 0, 128): 5,  # reefs
        }

        f_str = "{}{}".format(frame_index, gt_ext)

        image_path = os.path.join(self.data_path, folder.replace('sequence', 'gt'), f_str)
        color = self.loader(image_path)
        label_array = np.array(color)

        # 创建一个与标签图像大小相同的单通道图像
        label_encoded = np.zeros(label_array.shape[:2], dtype=np.uint8)

        # 使用向量化操作将 RGB 映射到单通道编码
        for color, label in color_to_label.items():
            mask = np.all(label_array == color, axis=-1)
            label_encoded[mask] = label
            # 处理未映射的颜色

        return Image.fromarray(label_encoded, mode='L')

        #return color

    def get_image(self, folder, frame_index, image_folder):
        f_str = "{}{}".format(frame_index, self.img_ext)

        image_path = os.path.join(
            self.data_path, folder, f_str)
        color = self.loader(image_path)

        return color

    def check_depth(self):
        line = self.filenames[0].split()
        # scene_name = line[0]
        scene_name = line[0].replace('sequence', '')
        frame_index = (line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "depth/{}.png".format((frame_index)))

        return os.path.isfile(velo_filename)

    def get_depth(self, folder, frame_index):
        folder1 = folder.replace('sequence', '')
        velo_filename = os.path.join(
            self.data_path,
            folder1,
            "depth/{}.png".format((frame_index)))

        depth_gt = Image.open(velo_filename)
        depth_gt = np.array(depth_gt)
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        return depth_gt

