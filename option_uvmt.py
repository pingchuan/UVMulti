from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in

def str2bool(v):
     if isinstance(v, bool):
          return v
     if v.lower() in ('yes', 'true', 't', 'y', '1'):
          return True
     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
          return False
     else:
          raise argparse.ArgumentTypeError('Boolean value expected.')

class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=r'/data2/lyx/data/')  #os.path.join(file_dir, "endovis_data")
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default='./logs_uvmt')
        self.parser.add_argument('--train_data_dir', type=str, default='train')
        self.parser.add_argument('--val_data_dir', type=str, default='test')
        self.parser.add_argument('--test_data_dir', type=str, default='test')
        # Model options
        self.parser.add_argument('--num_class', type=int, default=6)

        self.parser.add_argument("--pretrained_path",
                                 type=str,
                                 help="pretrained weights path",
                                 default=os.path.join(file_dir, "pretrained_model"))
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0])
        self.parser.add_argument("--warm_up_step",
                                 type=int,
                                 help="warm up step",
                                 default=4000)
        self.parser.add_argument("--residual_block_indexes",
                                 nargs="*",
                                 type=int,
                                 help="indexes for residual blocks in vitendodepth encoder",
                                 default=[2,5,8,11])
        self.parser.add_argument("--include_cls_token",
                                 type=str2bool,
                                 help="includes the cls token in the transformer blocks",
                                 default=True)
        self.parser.add_argument("--learn_intrinsics",
                                 type=str2bool,
                                 help="learn the camera intrinsics with a seperate decoder",
                                 default=False)
        self.parser.add_argument('--use_future_frame',
                                 action='store_true',
                                 help='If set, will also use a future frame in time for matching.')
        self.parser.add_argument('--num_matching_frames',
                                 help='Sets how many previous frames to load to build the cost'
                                      'volume',
                                 type=int,
                                 default=1)
        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="benthic")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["benthic,benthic_m"],
                                 default="benthic")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])

        self.parser.add_argument("--num_layers_depth",
                                 type=int,
                                 help="number of resnet layers",
                                 default=34,
                                 choices=[18, 34, 50, 101, 152])

        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="benthic",
                                 choices=["benthic"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=320)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=320)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--position_smoothness",
                                 type=float,
                                 help="registration smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--transform_constraint",
                                 type=float,
                                 help="transform constraint weight",
                                 default=0.01)
        self.parser.add_argument("--transform_smoothness",
                                 type=float,
                                 help="transform smoothness weight",
                                 default=0.01)

        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.001) #0.1,0.001
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=80.0) #40.0 #150.0
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=8)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=7e-5) #1e-4#5e-4/6e-5
        self.parser.add_argument("--learning_rate_pose",
                                 type=float,
                                 help="learning rate pose",
                                 default=1e-5)    #2e-5 /1e-5
        self.parser.add_argument("--learning_rate_depth",
                                 type=float,
                                 help="learning rate pose",
                                 default=7e-5) #1e-4 ,6e-5
        self.parser.add_argument('--power', type=float, default=0.85)#0.9,0.85
        self.parser.add_argument("--dim_out",
                                 type=int,
                                 help="number of bins",
                                 default=128)
        self.parser.add_argument("--query_nums",
                                 type=int,
                                 help="number of queries, should be less than h*w/p^2",
                                 default=128)
        self.parser.add_argument("--patch_size",
                                 type=int,
                                 help="patch size before ViT",
                                 default=32) #16，20
        self.parser.add_argument("--model_dim",
                                 type=int,
                                 help="model dim",
                                 default=32)
        # OPTIMIZATION options
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--max_steps",
                                 type=int,
                                 help="number of epochs",
                                 default=100000)#50000,30000,100000
        self.parser.add_argument("--eval_interval",
                                 type=int,
                                 help="number of eval",
                                 default=1000)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=20)

        # ABLATION options
        self.parser.add_argument("--disable_motion_masking",
                                 help="If set, will not apply consistency loss in regions where"
                                      "the cost volume is deemed untrustworthy",
                                 action="store_true")
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 default=False,
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",default=False,
                                 help="if set, uses average reprojection loss",
                                )
        self.parser.add_argument("--disable_automasking",default=True,
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",default=False,
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=0)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["position_encoder", "position"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=200)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--model_type",
                                 type=str,
                                 help="which training split to use",
                                 choices=["endodac", "afsfm"],
                                 default="endodac")
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="benthic",
                                 choices=[
                                    "hamlyn", "c3vd", "endovis", "canyons"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--visualize_depth",
                                 help="if set saves visualized depth map",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

        # EVALUATION options
        self.parser.add_argument("--save_recon",
                                 help="if set saves reconstruction files",
                                 action="store_true")
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
