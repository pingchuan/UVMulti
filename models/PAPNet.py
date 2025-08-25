import torch
import torch.nn as nn
import torch.nn.functional as F
#from models.resnet import Bottleneck
#from models.layers import SEBlock, SABlock
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


class SABlock(nn.Module):
    """ Spatial self-attention block """
    def __init__(self, in_channels, out_channels):
        super(SABlock, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                        nn.Sigmoid())
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        attention_mask = self.attention(x)
        features = self.conv(x)
        return torch.mul(features, attention_mask)
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

def compute_affinity_matrix(features):
    """
    Compute the affinity matrix for the given features.

    Args:
    - features (torch.Tensor): feature maps of shape (B, C, H, W)

    Returns:
    - affinity_matrix (torch.Tensor): affinity matrix of shape (B, H*W, H*W)
    """

    B, C, H, W = features.size()

    # Reshape features to (B, H*W, C)
    features = features.view(B, C, -1).transpose(1, 2)

    # Normalize the features
    features = F.normalize(features, dim=1)

    # Compute the dot product to get the affinity matrix
    affinity_matrix = torch.bmm(features, features.transpose(1, 2))

    # Row normalization
    affinity_matrix /= affinity_matrix.sum(1, keepdim=True)

    return affinity_matrix


def diffuse(features, joint_affinity, beta=0.05, num_iterations=3):
    """
    Apply the diffusion module to propagate patterns using the joint affinity matrix iteratively.

    Args:
    - features (torch.Tensor): Feature maps of shape (B, C, H, W)
    - joint_affinity (torch.Tensor): Joint affinity matrix of shape (B, H*W, H*W)
    - beta (float): Diffusion parameter, it determines the strength of propagation
    - num_iterations (int): Number of times to apply the diffusion process iteratively

    Returns:
    - diffused_features (torch.Tensor): Diffused feature maps of shape (B, C, H, W)
    """

    B, C, H, W = features.size()

    # Reshape features to (B, H*W, C)
    features_reshaped = features.view(B, C, -1).transpose(1, 2)

    diffused_features = features_reshaped.clone()

    # Iteratively apply the diffusion process
    for _ in range(num_iterations):
        diffused_features = beta * torch.bmm(joint_affinity, diffused_features) + (1 - beta) * features_reshaped

    # Reshape diffused features back to (B, C, H, W)
    diffused_features = diffused_features.transpose(1, 2).view(B, C, H, W)

    return diffused_features


class InitialTaskPredictionModule(nn.Module):
    """
        Make the initial task predictions from the backbone features.
    """

    def __init__(self, out_channels, tasks, input_channels, intermediate_channels=256):
        super(InitialTaskPredictionModule, self).__init__()
        self.tasks = tasks
        layers = {}
        conv_out = {}

        for task in self.tasks:
            if input_channels != intermediate_channels:
                downsample = nn.Sequential(nn.Conv2d(input_channels, intermediate_channels, kernel_size=1,
                                                     stride=1, bias=False), nn.BatchNorm2d(intermediate_channels))
            else:
                downsample = None
            bottleneck1 = Bottleneck(input_channels, intermediate_channels // 4, downsample=downsample)
            bottleneck2 = Bottleneck(intermediate_channels, intermediate_channels // 4, downsample=None)
            conv_out_ = nn.Conv2d(intermediate_channels, out_channels[task], 1)
            layers[task] = nn.Sequential(bottleneck1, bottleneck2)
            conv_out[task] = conv_out_

        self.layers = nn.ModuleDict(layers)
        self.conv_out = nn.ModuleDict(conv_out)

    def forward(self, x):
        out = {}

        for task in self.tasks:
            out['features_%s' % (task)] = self.layers[task](x)
            out[task] = self.conv_out[task](out['features_%s' % (task)])

        return out


class AffinityLearningModule(nn.Module):
    def __init__(self, auxilary_tasks, channels):
        super(AffinityLearningModule, self).__init__()
        self.tasks = auxilary_tasks

        # Conv layer to reduce channels from 2C to C
        self.conv = nn.ModuleDict({task: nn.Conv2d(channels, channels // 2, 1) for task in self.tasks})

    def forward(self, x, mat_size):
        out = {}
        for task in self.tasks:
            features = self.conv[task](x[f'features_{task}'])
            features = F.interpolate(features, mat_size, mode='bilinear')
            out[f'features_{task}'] = features
            out[f'matrix_{task}'] = compute_affinity_matrix(features)
        return out


class MultiTaskDiffusionModule(nn.Module):
    def __init__(self, tasks, auxilary_tasks, beta, num_iters):
        super(MultiTaskDiffusionModule, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.beta = beta
        self.num_iters = num_iters
        self.weights = nn.ParameterDict({t: nn.Parameter(torch.ones(len(tasks)) / len(tasks)) for t in self.tasks})

    # TODO: Impliment subsampling
    def forward(self, x):
        out = {}

        for task in self.tasks:
            # compute combined matrices
            task_weights = F.softmax(self.weights[task], dim=0)  # make sure weights sum up to 1
            combined_affinity_matrix = sum([w * x[f'matrix_{t}'] for t, w in zip(self.tasks, task_weights)])

            # diffuse
            features = x[f'features_{task}']
            out[task] = diffuse(features, combined_affinity_matrix, self.beta, self.num_iters)

        return out


class PAPNet(nn.Module):
    def __init__(self):
        super(PAPNet, self).__init__()
        # General
        #self.tasks = opt.TASKS.NAMES
        self.tasks = ['seg', 'enh', 'depth']
        self.auxilary_tasks = ['seg', 'enh', 'depth']
        #self.channels = [64,128,256,512]
        self.channels = 512
        self.beta = 0.05
        self.num_iters = 1
        #self.out_channels = opt.TASKS.NUM_OUTPUT
        self.out_channels = {'seg': 6, 'enh': 3, 'depth': 1}
        # Backbone


        # Task-specific heads for initial prediction
        self.initial_task_prediction_heads = InitialTaskPredictionModule(self.out_channels, self.auxilary_tasks,
                                                                         self.channels)

        # Cross-task proagation
        self.affinity_learning_module = AffinityLearningModule(self.auxilary_tasks, 256)

        # Task-specific propagation
        self.diffusion_module = MultiTaskDiffusionModule(self.tasks, self.auxilary_tasks, self.beta, self.num_iters)

        # Task-specific heads for final prediction
        heads = {}
        for task in self.tasks:
            bottleneck1 = Bottleneck(128, 128 // 4, downsample=None)
            bottleneck2 = Bottleneck(128, 128 // 4, downsample=None)
            conv_out_ = nn.Conv2d(128, self.out_channels[task], 1)
            heads[task] = nn.Sequential(bottleneck1, bottleneck2, conv_out_)

        self.heads = nn.ModuleDict(heads)

    def forward(self, x):
        self.outputs = {}
        out = {}

        # Backbone


        # Initial predictions for every task including auxilary tasks
        x = self.initial_task_prediction_heads(x[4])
        for task in self.auxilary_tasks:
            #self.outputs[task] = F.interpolate(x[task], [320,320], mode='bilinear')
            new_task_name = task + '0'
            self.outputs[new_task_name] = F.interpolate(x[task], [320, 320], mode='bilinear')

        self.outputs[('disp0', 0)] = torch.sigmoid(self.outputs['depth0'])
        # Affinty learning
        x = self.affinity_learning_module(x, mat_size=(320 // 4, 320 // 4))

        # Propagate patterns using join affinities
        x = self.diffusion_module(x)

        # Make final prediction with task-specific heads
        for task in self.tasks:
            self.outputs[task] = F.interpolate(self.heads[task](x[task]), [320,320], mode='bilinear')
        self.outputs[('disp', 0)] = torch.sigmoid(self.outputs['depth'])
        return self.outputs