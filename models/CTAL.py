import torch
import torch.nn as nn
import torch.nn.functional as F
#from models.resnet import BasicBlock, Bottleneck
#from models.layers import SEBlock
#from models.hrnet import HighResolutionFuse


class HighResolutionFuse(nn.Module):
    def __init__(self, backbone_channels, num_outputs):
        super(HighResolutionFuse, self).__init__()
        last_inp_channels = sum(backbone_channels)
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=0.1),
            nn.ReLU(inplace=False))

    def forward(self, x):
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], (x0_h, x0_w), mode='bilinear')
        x2 = F.interpolate(x[2], (x0_h, x0_w), mode='bilinear')
        x3 = F.interpolate(x[3], (x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.last_layer(x)
        return x
class SEBlock(nn.Module):
    """ Squeeze-and-excitation block """
    def __init__(self, channels, r=16):
        super(SEBlock, self).__init__()
        self.r = r
        self.squeeze = nn.Sequential(nn.Linear(channels, channels//self.r),
                                     nn.ReLU(),
                                     nn.Linear(channels//self.r, channels),
                                     nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.size()
        squeeze = self.squeeze(torch.mean(x, dim=(2,3))).view(B,C,1,1)
        return torch.mul(x, squeeze)
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class InitialTaskPredictionModule(nn.Module):
    """ Module to make the inital task predictions """

    def __init__(self, auxilary_tasks, input_channels, task_channels):
        super(InitialTaskPredictionModule, self).__init__()
        self.auxilary_tasks = auxilary_tasks
        NUM_OUTPUT = {'seg': 6, 'enh': 3, 'depth': 1}
        # Per task feature refinement + decoding
        if input_channels == task_channels:
            channels = input_channels
            self.refinement = nn.ModuleDict(
                {task: nn.Sequential(BasicBlock(channels, channels), BasicBlock(channels, channels)) for task in
                 self.auxilary_tasks})

        else:
            refinement = {}
            for t in auxilary_tasks:
                downsample = nn.Sequential(nn.Conv2d(input_channels, task_channels, 1, bias=False),
                                           nn.BatchNorm2d(task_channels))
                refinement[t] = nn.Sequential(BasicBlock(input_channels, task_channels, downsample=downsample),
                                              BasicBlock(task_channels, task_channels))
            self.refinement = nn.ModuleDict(refinement)

        self.decoders = nn.ModuleDict(
            {task: nn.Conv2d(task_channels, NUM_OUTPUT[task], 1) for task in self.auxilary_tasks})

    def forward(self, features_curr_scale, features_prev_scale=None):
        if features_prev_scale is not None:  # Concat features that were propagated from previous scale
            x = {t: torch.cat(
                (features_curr_scale, F.interpolate(features_prev_scale[t], scale_factor=2, mode='bilinear')), 1) for t
                 in self.auxilary_tasks}

        else:
            x = {t: features_curr_scale for t in self.auxilary_tasks}

        # Refinement + Decoding
        out = {}
        for t in self.auxilary_tasks:
            out['features_%s' % (t)] = self.refinement[t](x[t])
            out[t] = self.decoders[t](out['features_%s' % (t)])

        return out


class FPM(nn.Module):
    """ Feature Propagation Module """

    def __init__(self, auxilary_tasks, per_task_channels):
        super(FPM, self).__init__()
        # General
        self.auxilary_tasks = auxilary_tasks
        self.N = len(self.auxilary_tasks)
        self.per_task_channels = per_task_channels
        self.shared_channels = int(self.N * per_task_channels)

        # Non-linear function f
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.shared_channels // 4, 1, bias=False),
                                   nn.BatchNorm2d(self.shared_channels // 4))
        self.non_linear = nn.Sequential(
            BasicBlock(self.shared_channels, self.shared_channels // 4, downsample=downsample),
            BasicBlock(self.shared_channels // 4, self.shared_channels // 4),
            nn.Conv2d(self.shared_channels // 4, self.shared_channels, 1))

        # Dimensionality reduction
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.per_task_channels, 1, bias=False),
                                   nn.BatchNorm2d(self.per_task_channels))
        self.dimensionality_reduction = BasicBlock(self.shared_channels, self.per_task_channels,
                                                   downsample=downsample)

        # SEBlock
        self.se = nn.ModuleDict({task: SEBlock(self.per_task_channels) for task in self.auxilary_tasks})

    def forward(self, x):
        # Get shared representation
        concat = torch.cat([x['features_%s' % (task)] for task in self.auxilary_tasks], 1)
        B, C, H, W = concat.size()
        shared = self.non_linear(concat)
        mask = F.softmax(shared.view(B, C // self.N, self.N, H, W), dim=2)  # Per task attention mask
        shared = torch.mul(mask, concat.view(B, C // self.N, self.N, H, W)).view(B, -1, H, W)

        # Perform dimensionality reduction
        shared = self.dimensionality_reduction(shared)

        # Per task squeeze-and-excitation
        out = {}
        for task in self.auxilary_tasks:
            out[task] = self.se[task](shared) + x['features_%s' % (task)]

        return out


class SpatialAttentionModule(nn.Module):
    def __init__(self, tasks, channels, feature_size, gamma):
        super(SpatialAttentionModule, self).__init__()
        self.tasks = tasks

        # Value projection for each task
        self.projection = nn.ModuleDict(
            {task: nn.Conv2d(channels, channels, kernel_size=3, padding=1) for task in self.tasks})

        # Cross-Task Affinity Learning Modules
        self.task_fusion = nn.Conv2d(in_channels=len(self.tasks) * feature_size, out_channels=feature_size,
                                     kernel_size=3, stride=1, padding=1, groups=feature_size)

        # Blending parameter
        self.gamma = gamma

    def forward(self, x, return_m=False):
        out = {}
        M = []
        B, C, H, W = list(x.values())[0].size()

        # compute all self-attention masks
        for task in self.tasks:

            features = x[f'features_{task}'].view(B, C, H * W)
            # Normalize the features
            features = F.normalize(features, dim=1)
            # Compute the inner product to get the affinity matrix
            affinity_matrix = torch.bmm(features.transpose(1, 2), features)
            # Reshape matrix to (B, H*W, H, W)
            affinity_matrix = affinity_matrix.view(B, -1, H, W)

            M.append(affinity_matrix)

        # channel-wise interleave concatenation
        M = torch.stack(M, dim=2).reshape(B, -1, H, W)

        M = self.task_fusion(M)

        # compute and apply all cross-attention masks
        for task in self.tasks:
            features = x[f'features_{task}']
            attention = M.view(B, H * W, H * W).transpose(1, 2)
            value = self.projection[task](features).view(B, C, -1)
            attended_features = torch.bmm(value, attention).view(B, -1, H, W)
            x_out = self.gamma * attended_features + (1 - self.gamma) * features
            out[f'features_{task}'] = x_out

        return out


class AffinityLearningModule(nn.Module):
    def __init__(self, auxilary_tasks, in_channels, out_channels, feature_size, gamma):
        super(AffinityLearningModule, self).__init__()
        self.tasks = auxilary_tasks

        self.conv_in = nn.ModuleDict({task: nn.Conv2d(in_channels, out_channels, 1) for task in self.tasks})

        # multitask spatial attention module
        self.spatial_att = SpatialAttentionModule(self.tasks, out_channels, feature_size, gamma)

        # channel reduction layers for outgoing features
        conv_out = {}
        for task in self.tasks:
            out = nn.Sequential(
                nn.ConvTranspose2d(out_channels, out_channels, 2, 2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                # nn.Dropout2d(0.1),
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            )
            conv_out[task] = out
        self.conv_out = nn.ModuleDict(conv_out)

    def forward(self, x):
        out = {}
        for task in self.tasks:
            x[f'features_{task}'] = self.conv_in[task](x[f'features_{task}'])
        x = self.spatial_att(x)
        for task in self.tasks:
            out[f'features_{task}'] = self.conv_out[task](x[f'features_{task}'])

        return out


class MEMANet(nn.Module):
    """
        MTI-Net implementation based on HRNet backbone
        https://arxiv.org/pdf/2001.06902.pdf
    """

    def __init__(self):
        super(MEMANet, self).__init__()
        # General
        self.tasks = ['seg', 'enh', 'depth']
        # self.auxilary_tasks = p.AUXILARY_TASKS.NAMES
        self.auxilary_tasks = ['seg', 'enh', 'depth']

        self.channels = [64, 128, 256, 512]
        self.num_scales = len(self.channels)
        backbone_dims=[64,64]
        self.embedding_size = torch.tensor(backbone_dims, dtype=torch.int32).prod().item()
        self.gamma = 0.05
        #self.out_channels = opt.TASKS.NUM_OUTPUT
        self.out_channels = {'seg': 6, 'enh': 3, 'depth': 1}
        intermediate_channels = 128

        # Backbone


        # Feature Propagation Module
        self.fpm_scale_3 = FPM(self.auxilary_tasks, self.channels[3])
        self.fpm_scale_2 = FPM(self.auxilary_tasks, self.channels[2])
        self.fpm_scale_1 = FPM(self.auxilary_tasks, self.channels[1])

        # Initial task predictions at multiple scales
        self.scale_0 = InitialTaskPredictionModule(self.auxilary_tasks, self.channels[0] + self.channels[1],
                                                   self.channels[0])
        self.scale_1 = InitialTaskPredictionModule(self.auxilary_tasks, self.channels[1] + self.channels[2],
                                                   self.channels[1])
        self.scale_2 = InitialTaskPredictionModule(self.auxilary_tasks, self.channels[2] + self.channels[3],
                                                   self.channels[2])
        self.scale_3 = InitialTaskPredictionModule(self.auxilary_tasks, self.channels[3], self.channels[3])

        # Initial predisction fusion
        self.scale_fusion = nn.ModuleDict({task: HighResolutionFuse(self.channels, 256) for task in self.tasks})

        # Distillation at multiple scales
        self.ctal = AffinityLearningModule(self.auxilary_tasks, sum(self.channels), intermediate_channels,
                                           self.embedding_size, self.gamma)

        # Task-specific heads for final prediction
        heads = {}
        for task in self.tasks:
            bottleneck1 = Bottleneck(intermediate_channels, intermediate_channels // 4, downsample=None)
            bottleneck2 = Bottleneck(intermediate_channels, intermediate_channels // 4, downsample=None)
            conv_out_ = nn.Conv2d(intermediate_channels, self.out_channels[task], 1)
            heads[task] = nn.Sequential(bottleneck1, bottleneck2, conv_out_)

        self.heads = nn.ModuleDict(heads)

    def forward(self, x):
        self.outputs = {}
        out = {}

        # Backbone

        # Predictions at multiple scales
        # Scale 3
        x_3 = self.scale_3(x[4])
        x_3_fpm = self.fpm_scale_3(x_3)
        # Scale 2
        x_2 = self.scale_2(x[3], x_3_fpm)
        x_2_fpm = self.fpm_scale_2(x_2)
        # Scale 1
        x_1 = self.scale_1(x[2], x_2_fpm)
        x_1_fpm = self.fpm_scale_1(x_1)
        # Scale 0
        x_0 = self.scale_0(x[1], x_1_fpm)

        out['deep_supervision'] = {'scale_0': x_0, 'scale_1': x_1, 'scale_2': x_2, 'scale_3': x_3}

        x_f = {}
        for task in self.tasks:
            x_f[f'features_{task}'] = self.scale_fusion[task](
                [x_0[f'features_{task}'], x_1[f'features_{task}'], x_2[f'features_{task}'], x_3[f'features_{task}']])
        x_out = self.ctal(x_f)

        # Feature aggregation
        for t in self.tasks:
            self.outputs[t] = F.interpolate(self.heads[t](x_out[f'features_{t}']), [256,256], mode='bilinear')

        self.outputs[('disp', 0)] = torch.sigmoid(self.outputs['depth'])
        return self.outputs


def concat(x, y, tasks):
    for t in tasks:
        x[f'features_{t}'] = torch.cat((x[f'features_{t}'], y[f'features_{t}']), dim=1)
    return x