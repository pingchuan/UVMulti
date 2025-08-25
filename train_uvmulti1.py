from __future__ import absolute_import, division, print_function
import os
import time
import json
import torch.nn as nn
import datasets
import cv2
from datasets.benthic_dataset_multitask2 import BenthicDataset
from models.resnet_encoder import ResnetEncoder
from datasets.benthic_dataset_multitask_cz import BenthicDataset_CZ
from models.depth_decoder import DepthDecoder
from models.UVMTNet import PoseDecoder, SegDecoder_f, EnhDecoder_f, Depth_h_bins, Seg_h, Depth_h, Enh_h, EncoderD4,EncoderD5,EncoderD6,EncoderD7
from models.monodepth_layers import SSIM, BackprojectDepth, Project3D, disp_to_depth, compute_errors, \
    transformation_from_parameters, get_smooth_loss, sec_to_hm_str, normalize_image
from models.utils.loss import BDiceLoss_sup, BceLoss1_sup,SCRLoss,FCDLoss
import numpy as np
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.utils.metrics_seg_enh import EnhancementMetrics, StreamSegMetrics
import numpy as np
import random
from option_uvmulti import MonodepthOptions

splits_dir = os.path.join(os.path.dirname(__file__), "splits")
options = MonodepthOptions()
opts = options.parse()

class AWA:
    def __init__(self, task_num, temperature=1.0, scale_factor=1.0, device='cuda'):
        self.task_num = task_num  # Number of tasks
        self.temperature = temperature  # Temperature factor for softmax scaling
        self.scale_factor = scale_factor  # Scaling factor for the final weights
        self.device = device


        self.loss_history = {i: [1.0, 1.0] for i in range(self.task_num)}

    def compute_learning_speed(self, losses):
        """
        Compute the learning speed for each task.
        The learning speed is the ratio of the current loss to the previous loss.
        """
        learning_speeds = []
        for i in range(self.task_num):
            # Calculate the learning speed: current loss / previous loss
            speed = self.loss_history[i][0] / self.loss_history[i][1]
            learning_speeds.append(speed)
        return learning_speeds

    def update_loss_history(self, losses):
        """
        Update the loss history with the current losses.
        Only update the first two elements of the history.
        """
        for i in range(self.task_num):
            # Update the previous loss history
            self.loss_history[i][1] = self.loss_history[i][0]
            # Update the current loss history
            self.loss_history[i][0] = losses[i].detach()

    def compute_task_weights(self, learning_speeds):
        """
        Compute task weights based on learning speeds.
        The weight for each task is computed using softmax with a temperature scaling.
        """
        # Scale the learning speeds using the temperature factor
        scaled_speeds = [speed / self.temperature for speed in learning_speeds]

        # Apply softmax to the scaled learning speeds
        exp_speeds = torch.exp(torch.tensor(scaled_speeds).to(self.device))
        softmax_weights = exp_speeds / torch.sum(exp_speeds)


        scaled_weights = self.scale_factor * softmax_weights
        return scaled_weights

    def compute_loss(self, losses, T=8000, gamma=0.9,step=2):
        """
        Compute the total loss considering task-specific weights and dynamic weight adjustments.
        """
        # Ensure the losses contain the required values
        assert "loss_depth" in losses, "loss_depth is missing in losses"
        assert "loss_seg" in losses, "loss_seg is missing in losses"
        assert "loss_enh" in losses, "loss_enh is missing in losses"

        # Update the loss history with the current losses
        self.update_loss_history([losses["loss_depth"], losses["loss_seg"], losses["loss_enh"]])

        # Compute the learning speeds for each task
        #learning_speeds = self.compute_learning_speed(losses)

        # First stage: if i < T, use fixed weights for each task
        i = step  # Assuming that losses include the current iteration number
        if i < T:
            lambda_s = lambda_d = lambda_e = 1.0
        else:
            # Second stage: dynamically adjust the weights
            # Compute the learning rate for the enhancement task (detach to prevent gradient computation)
            loss_enh = losses["loss_enh"].detach()
            loss_enh_prev = self.loss_history[2][1]
            r_enh = (gamma * loss_enh_prev + (1 - gamma) * loss_enh.detach()) / loss_enh

            # Calculate the relative magnitudes of the segmentation and depth tasks (detach to prevent gradient computation)
            loss_seg = losses["loss_seg"].detach()
            loss_depth = losses["loss_depth"].detach()
            m_s = loss_seg / (loss_seg + loss_depth)
            m_d = loss_depth / (loss_seg + loss_depth)

            # Compute the weight for the enhancement task
            lambda_e = 1 - torch.exp(-1 / r_enh)

            # Calculate the weights for segmentation and depth tasks based on their relative magnitudes
            M_i = 3 - lambda_e  # Ensuring the sum of all weights equals the number of tasks
            lambda_s = M_i * torch.exp(1 / m_s) / (torch.exp(1 / m_s) + torch.exp(1 / m_d))
            lambda_d = M_i * torch.exp(1 / m_d) / (torch.exp(1 / m_s) + torch.exp(1 / m_d))

        # Compute the final weighted loss
        weighted_loss = (lambda_s * losses["loss_seg"] +
                         lambda_d * losses["loss_depth"] +
                         lambda_e * losses["loss_enh"])

        return weighted_loss, (lambda_s, lambda_d, lambda_e)
def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def set_seed(inc, base_seed=8888):
    # cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    seed = base_seed + inc
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    os.environ['PYTHONHASHSEED'] = str(seed + 4)


class L2Loss_rgb(nn.Module):
    def __init__(self, lambda_rgb=1.0):
        super(L2Loss_rgb, self).__init__()
        self.lambda_rgb = lambda_rgb

    def forward(self, pred, target):

        l2_loss = F.mse_loss(pred, target, reduction='mean')


        loss = self.lambda_rgb * l2_loss

        return loss


class Trainer:
    def __init__(self, options):
        set_seed(1024)#8888
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []
        self.parameters_to_train_0 = []
        self.parameters_depth = []
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        ###loss

        self.seg_loss = BDiceLoss_sup(numclass=self.opt.num_class).cuda()
        self.enh_loss = L2Loss_rgb().cuda()

        self.fcd = FCDLoss().cuda()
        self.awa = AWA(task_num=3, temperature=1.0, scale_factor=1.0, device='cuda')
        ###metrics
        self.metrics_seg = StreamSegMetrics(num_classes=self.opt.num_class)
        self.metrics_enh = EnhancementMetrics()

        self.num_scales = len(self.opt.scales)  # 4
        self.num_input_frames = len(self.opt.frame_ids)  # 3
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames  # 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])


        frames_to_load = self.opt.frame_ids.copy()
        self.matching_ids = [0]
        if self.opt.use_future_frame:
            self.matching_ids.append(1)
        for idx in range(-1, -1 - self.opt.num_matching_frames, -1):
            self.matching_ids.append(idx)
            if idx not in frames_to_load:
                frames_to_load.append(idx)

        print('Loading frames: {}'.format(frames_to_load))

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")


        self.models["encoder"] = EncoderD7(
            num_layers=self.opt.num_layers_depth, pretrained=True, num_classes=self.opt.num_class)
        self.models["encoder"].to(self.device)
        self.parameters_to_train = list(self.models["encoder"].parameters())

        self.models["segmentation"] = Seg_h(
            num_classes=self.opt.num_class)
        self.models["segmentation"].to(self.device)
        self.parameters_to_train += list(self.models["segmentation"].parameters())

        self.models["enhancement"] = Enh_h(
            num_classes=self.opt.num_class)
        self.models["enhancement"].to(self.device)
        self.parameters_to_train += list(self.models["enhancement"].parameters())
        self.pose_params = []

        self.models["depth"] = Depth_h(num_classes=self.opt.num_class)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:

            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = ResnetEncoder(
                    self.opt.num_layers,
                    pretrained=True,
                    num_input_images=self.num_pose_frames)
                self.models["pose_encoder"].to(self.device)

                self.pose_params += list(self.models["pose_encoder"].parameters())
                self.models["pose"] = PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            self.models["pose"].to(self.device)

            self.pose_params += list(self.models["pose"].parameters())
        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"


            self.models["predictive_mask"] = DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())
        df_params = [{"params": self.pose_params, "lr": self.opt.learning_rate_pose},
                     {"params": self.parameters_to_train, "lr": self.opt.learning_rate}]

        self.model_optimizer = optim.Adam(df_params, lr=self.opt.learning_rate)

        self.model_lr_scheduler = optim.lr_scheduler.LambdaLR(self.model_optimizer,
                           lambda e: 1.0 - pow((e / self.opt.max_steps), self.opt.power))  #

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        datasets_dict = {"benthic": BenthicDataset_CZ}
        dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files2.txt")
        train_filenames = readlines(fpath.format("train"))
        test_filenames = readlines(fpath.format("test"))
        img_ext = '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = self.opt.max_steps

        train_dataset = dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, len(self.opt.scales), is_train=True, load_gt=True, load_enh_img=True, img_ext=img_ext,
            data2_dir=self.opt.train_data_dir)

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=0, pin_memory=True, drop_last=True)
        test_dataset = dataset(
            self.opt.data_path, test_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, len(self.opt.scales), load_depth=True, load_gt=True, load_enh_img=True,
            data2_dir=self.opt.test_data_dir)
        self.test_loader = DataLoader(
            test_dataset, 1, False,
            num_workers=0, pin_memory=True, drop_last=True)
        self.test_iter = iter(self.test_loader)

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}

        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rmse", "de/log_rmse", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items,  {:d} testing items\n".format(
            len(train_dataset), len(test_dataset)))

        self.save_opts()
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0
        for name, param in self.models["encoder"].named_parameters():


            mulValue = np.prod(param.size())
            Total_params += mulValue
            if param.requires_grad == False:
                NonTrainable_params += mulValue

            else:
                Trainable_params += mulValue

        print(f'Total params: {Total_params}')
        print(f'Trainable params: {Trainable_params}')

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        #lr 比例
        self.bl1 = 1
        self.bl2 = 1
        self.bl3 = 1
        self.step = 0
        self.best_rmse = float('inf')
        self.best_a1 = 0
        max_steps = self.opt.max_steps
        self.start_time = time.time()
        while self.step <= max_steps:

            self.run_epoch_multi()


        self.save_model(mode='final')

    def run_epoch_multi(self):
        """Run a single epoch of training and validation
        """

        print("Training")

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            self.set_train()
            outputs, losses = self.process_batch_multi(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            phase = batch_idx % self.opt.log_frequency == 0

            if phase or batch_idx == 10:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data, losses["loss_seg"].cpu().data, losses["loss_enh"].cpu().data)

            if (self.step % self.opt.eval_interval == 0 and self.step > 0) or self.step == 10:

                rmse, a1, miou, psnr = self.run_epoch_eval_multi()


                if rmse < self.best_rmse:
                    self.best_rmse = rmse
                    self.best_a1 = a1
                    self.save_model(mode='iter')


                print(f"Step {self.step} | RMSE: {rmse:.4f} | A1: {a1:.4f} | mIoU: {miou:.4f} | PSNR: {psnr:.4f}")
            self.model_lr_scheduler.step(self.step)
            self.step += 1



    def run_epoch_eval_multi(self):
        """Run a single epoch of evaluation
        """

        print("Evaluating")
        MIN_DEPTH = self.opt.min_depth  # 1e-3
        MAX_DEPTH = self.opt.max_depth  # 150,40

        self.set_eval()
        pred_depths = []

        gt_depths = []


        for batch_idx, inputs in enumerate(self.test_loader):
            with torch.no_grad():
                input_color = inputs[("color", 0, 0)].cuda()

                if self.opt.post_process:

                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)


                feats1, _ = self.models["encoder"](input_color)

                pred_enhs = self.models["enhancement"](feats1)
                pred_enh = pred_enhs["enh"].cpu().detach().numpy()
                # print(pred_enh.shape,inputs["enh"].cpu().numpy().shape)
                pred_seg = self.models["segmentation"](feats1)
                pred_seg = pred_seg["seg"].argmax(dim=1)
                # print(pred_seg.cpu().numpy().shape, inputs["gt"].cpu().numpy().shape)
                self.metrics_seg.update(pred_seg.cpu().detach().numpy(), inputs["gt"].cpu().numpy())
                # pred_enhs.append(pred_enh)

                self.metrics_enh.update(pred_enh, inputs["enh"].detach().cpu().numpy())

                output_depth = self.models["depth"](feats1)
                _, pred_depth = disp_to_depth(output_depth[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
                # pred_depth = output_depth[("disp", 0)]
                pred_depth = torch.clamp(F.interpolate(
                    pred_depth, [720, 1280], mode="bilinear", align_corners=True), MIN_DEPTH, MAX_DEPTH)  # false 改了true
                pred_depth = pred_depth[:, 0].cpu().detach().numpy()

                gt_depths.append(inputs["depth_gt"].squeeze().cpu().detach().numpy())
                pred_depths.append(pred_depth)

        results_enh = self.metrics_enh.get_results()
        self.metrics_enh.reset()
        results_seg = self.metrics_seg.get_results()
        self.metrics_seg.reset()

        pred_depths = np.concatenate(pred_depths)

        errors = []
        ratios = []
        # import matplotlib

        # cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        import matplotlib.cm as cm

        cmap = cm.get_cmap('Spectral_r')
        for i in range(pred_depths.shape[0]):
            gt_depth = gt_depths[i]
            # gt_height, gt_width = gt_depth.shape[:2]
            ##
            pred_depth = pred_depths[i]

            depth = np.clip(pred_depth, 0, 80)
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            # depth = np.clip(pred_depth, 0, 80)  # 保证深度在 [0, 100] 范围内
            depth = depth.astype(np.uint8)
            colored_depth = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)


            save_path = os.path.join(self.log_path, "pred", "{:05d}.png".format(i))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # cv2.imwrite(save_path, depth_color_mapped)
            cv2.imwrite(save_path, colored_depth)

            # pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))

            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            if np.all(mask == False):
                #print(f"Skipping image {i} because no valid depth values are within the range.")
                continue
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            pred_depth *= self.opt.pred_depth_scale_factor

            # print(pred_depth.max(), pred_depth.min())
            if not self.opt.disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            errors.append(compute_errors(gt_depth, pred_depth))
        if not self.opt.disable_median_scaling:
            ratios = np.array(ratios)
            med = np.median(ratios)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

        mean_errors = np.array(errors).mean(0)
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print(
            f"Iter {self.step}/{self.opt.max_steps} - Valid Seg: {', '.join([f'{k}:{v :.5f}' for k, v in results_seg.items()])}")
        print()  # 这个空的 print 语句会输出一个换行符，确保两行分开
        print(
            f"Iter {self.step}/{self.opt.max_steps} - Valid Enh: {', '.join([f'{k}:{v :.5f}' for k, v in results_enh.items()])}")
        self.set_train()

        return mean_errors[2], mean_errors[4], results_seg['mean_IoU'], results_enh['mean_PSNR']



    def process_batch_multi(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        # outputs = self.models["depth_encoder"](inputs["color_aug", 0, 0])
        outputs = {}
        # feats = self.models["encoder"](inputs["color_aug", 0, 0])
        feats,out1 = self.models["encoder"](inputs["color", 0, 0])
        outputs.update(out1)
        outputs["feats"] = feats
        outputs.update(self.models['enhancement'](feats))
        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](feats)
        outputs.update(self.models['depth'](feats))
        ##segmentation,
        outputs.update(self.models['segmentation'](feats))
        ## enhancement

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, feats))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs


    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp0", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)  # 改了
                source_scale = 0
                # depth = disp
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            # depth = disp
            outputs[("depth0", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color0", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity0", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)  # 改了
                source_scale = 0
                # depth = disp
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            # depth = disp
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):

        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss_d0 = 0
        total_loss_d = 0
        total_loss_d_1 = 0
        total_loss = 0
        #e,s,d = outputs["feats"]
        #losses["loss_fcd"] = (self.fcd(outputs["fe"],e) + self.fcd(outputs["fs"],s)+self.fcd(outputs["fd"],d))/3
        losses["loss_enh"] = self.enh_loss(outputs["enh"], inputs["enh"].squeeze(1).float())
        losses["loss_seg"] = self.seg_loss(outputs["seg"], inputs["gt"].squeeze(1).long())
        #losses["loss_enh0"] = self.enh_loss(outputs["enh0"], inputs["enh"].squeeze(1).float())
        losses["loss_seg0"] = self.seg_loss(outputs["seg0"], inputs["gt"].squeeze(1).long())
        # losses["loss_enh_1"] = self.enh_loss(outputs["enh_1"], inputs["enh"].squeeze(1).float())
        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp0", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color0", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                        idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss_d0 += loss
            losses["loss0/{}".format(scale)] = loss

        total_loss_d0 /= self.num_scales


        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                        idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss_d += loss
            losses["loss/{}".format(scale)] = loss

        total_loss_d /= self.num_scales
        losses["loss_depth"] = total_loss_d
        losses["loss1"], _ = self.awa.compute_loss(losses, step=self.step)
        #losses["loss"] = (total_loss_d/total_loss_d.detach() + losses["loss_seg"]/losses["loss_seg"].detach() + losses["loss_enh"]/losses["loss_enh"].detach())
        #losses["loss"] = ((total_loss_d + losses["loss_seg"] + losses["loss_enh"])+(total_loss_d0 + losses["loss_seg0"] + losses["loss_enh0"]))/2
        losses["loss"] = ((losses["loss1"]) + (
                    total_loss_d0 + losses["loss_seg0"]))


        return losses

    def compute_loss_masks(self, reprojection_loss, identity_reprojection_loss):
        """ Compute loss masks for each of standard reprojection and depth hint
        reprojection"""

        if identity_reprojection_loss is None:
            # we are not using automasking - standard reprojection loss applied to all pixels
            reprojection_loss_mask = torch.ones_like(reprojection_loss)

        else:
            # we are using automasking
            all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
            idxs = torch.argmin(all_losses, dim=1, keepdim=True)
            reprojection_loss_mask = (idxs == 0).float()

        return reprojection_loss_mask

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        # with torch.no_grad():
        #   outputs, losses = self.process_batch_val(inputs)
        #  ##self.log("val", inputs, outputs, losses)
        # del inputs, outputs, losses

        self.set_train()



    def log_time(self, batch_idx, duration, loss,loss_seg,loss_enh):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "step {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | loss_s: {:.5f} | loss_e: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.step, batch_idx, samples_per_sec, loss,loss_seg,loss_enh,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids[1:]:

                    writer.add_image(
                        "brightness_{}_{}/{}".format(frame_id, s, j),
                        outputs[("transform", "high", s, frame_id)][j].data, self.step)
                    writer.add_image(
                        "registration_{}_{}/{}".format(frame_id, s, j),
                        outputs[("registration", s, frame_id)][j].data, self.step)
                    writer.add_image(
                        "refined_{}_{}/{}".format(frame_id, s, j),
                        outputs[("refined", s, frame_id)][j].data, self.step)
                    if s == 0:
                        writer.add_image(
                            "occu_mask_backward_{}_{}/{}".format(frame_id, s, j),
                            outputs[("occu_mask_backward", s, frame_id)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, mode='iter'):
        """Save model weights to disk
        """
        if mode == 'iter':
            save_folder = os.path.join(self.log_path, "models", "weights_0")
        elif mode == 'last':
            save_folder = os.path.join(self.log_path, "models", "weights_last")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'depth':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        print("Adam is randomly initialized")


if __name__ == "__main__":
    # random_seeds(314)
    trainer = Trainer(opts)
    trainer.train()