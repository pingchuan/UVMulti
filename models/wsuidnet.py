import torch
import torch.nn as nn
import torch.nn.functional as F


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


class BalancedFusionModule(nn.Module):
    """ CWF：Implementation of Complementary Weighted Fusion Module  """
    def __init__(self, in_channels, out_channels):
        super(BalancedFusionModule, self).__init__()

        self.attention = nn.Sequential(nn.Conv2d(in_channels*2, out_channels, 3, padding=1, bias=False),
                                       nn.Sigmoid())
        self.conv_2 = nn.Conv2d(in_channels*2, out_channels, 3, padding=1, bias=False)

    def forward(self, x, y):
        F = torch.cat((x, y), dim=1)
        attention_mask = self.attention(F)

        features_x = torch.mul(x, attention_mask)
        features_y = torch.mul(y, torch.ones_like(attention_mask) - attention_mask)
        out_x = features_x + x
        out_y = features_y + y

        out = torch.cat((out_x, out_y), dim=1)
        out = self.conv_2(out)
        return out
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
    """
        Make the initial task predictions from the backbone features.
    """

    def __init__(self,  TASKS, input_channels, intermediate_channels=256):
        super(InitialTaskPredictionModule, self).__init__()
        self.tasks = ['seg', 'enh', 'depth']
        layers = {}
        conv_out = {}
        NUM_OUTPUT = {'seg': 6, 'enh': 3, 'depth': 1}
        for task in self.tasks:
            if input_channels != intermediate_channels:
                downsample = nn.Sequential(nn.Conv2d(input_channels, intermediate_channels, kernel_size=1,
                                                     stride=1, bias=False), nn.BatchNorm2d(intermediate_channels))
            else:
                downsample = None
            bottleneck1 = Bottleneck(input_channels, intermediate_channels // 4, downsample=downsample)
            # upsampling = nn.Upsample(scale_factor=2)
            bottleneck2 = Bottleneck(intermediate_channels, intermediate_channels // 4, downsample=None)
            conv_out_ = nn.Conv2d(intermediate_channels,  NUM_OUTPUT[task], 1)
            layers[task] = nn.Sequential(bottleneck1, bottleneck2)
            conv_out[task] = nn.Sequential(conv_out_)

        self.layers = nn.ModuleDict(layers)
        self.conv_out = nn.ModuleDict(conv_out)

    def forward(self, x):
        out = {}

        for task in self.tasks:
            out['features_%s' % (task)] = self.layers[task](x)
            out[task] = self.conv_out[task](out['features_%s' % (task)])

        return out


class Multi_feature_Fusion_Module(nn.Module):
    """
        Perform Multi-Task Distillation
        We apply an attention mask to features from other tasks and
        add the result as a residual.
    """

    def __init__(self, tasks, auxilary_tasks, channels):
        super(Multi_feature_Fusion_Module, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.BalancedFusionModule = BalancedFusionModule(channels, channels)

    def forward(self, x):
        out = {}
        for t in self.tasks:
            other_tasks = [a for a in self.auxilary_tasks if a != t]
            other_task_features = self.BalancedFusionModule(x['features_%s' % (other_tasks[0])], x['features_%s' % (other_tasks[1])])
            out[t] = x['features_%s' % (t)] + other_task_features
        return out
from models.resnet_encoder import ResnetEncoder


class UDE_Net(nn.Module):
    def __init__(self, num_layers, pretrained, num_classes):
        super(UDE_Net, self).__init__()
        # General
        self.tasks = ['seg', 'enh', 'depth']
        self.auxilary_tasks = ['seg', 'enh', 'depth']
        self.channels = 512
        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained)
        # Backbone
        # self.backbone = backbone
        NUM_OUTPUT = {'seg': 6, 'enh': 3, 'depth': 1}
        # Task-specific heads for initial prediction
        self.initial_task_prediction_heads = InitialTaskPredictionModule(self.auxilary_tasks,
                                                                         self.channels)
        # Multi-modal distillation
        self.Multi_feature_Fusion_Module = Multi_feature_Fusion_Module(self.tasks, self.tasks, 256)
        # Task-specific heads for  prediction
        heads = {}
        for task in self.auxilary_tasks:
            bottleneck1 = Bottleneck(256, 256 // 4, downsample=None)
            bottleneck2 = Bottleneck(256, 256 // 4, downsample=None)
            # upsampling = nn.Upsample(scale_factor=2)
            conv_out_ = nn.Conv2d(256, NUM_OUTPUT[task], 1)
            heads[task] = nn.Sequential(bottleneck1, bottleneck2, conv_out_)

        self.heads = nn.ModuleDict(heads)
        # self.sigmoid = nn.Sigmoid()
        self.downsample = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False),
                                        nn.BatchNorm2d(64))

        self.conv1x1 = nn.Conv2d(in_channels=320,
                                 out_channels=256, kernel_size=1, stride=1,
                                 padding=0, bias=True)

        self.channel_attention = SEBlock(channels=320, r=16)

    def forward(self, x):
        img_size = x.size()[-2:]
        out = {}
        self.outputs = {}
        features = {}

        # Backbone
        _, _, _, _, x = self.encoder(x)

        # Initial predictions for every task including auxilary tasks
        x = self.initial_task_prediction_heads(x)
        for task in self.auxilary_tasks:
            out['initial_%s' % (task)] = x[task]
            features['initial_features_%s' % (task)] = self.downsample(x['features_%s' % (task)])

        # Refine features through multi-modal distillation
        x = self.Multi_feature_Fusion_Module(x)

        for task in self.tasks:
            out['middle_%s' % (task)] = F.interpolate(self.heads[task](x[task]), img_size, mode='bilinear')
            self.outputs[task] = F.interpolate(self.heads[task](x[task]), img_size, mode='bilinear')
        multi_features = torch.cat((features['initial_features_enh'],
                                    features['initial_features_depth'],
                                    self.downsample(x['enh']),
                                    self.downsample(x['depth']),
                                    self.downsample(x['seg'])), dim=1)

        multi_features = self.channel_attention(multi_features) + multi_features
        multi_features = self.conv1x1(multi_features)
        depth = F.interpolate(self.heads['depth'](multi_features), img_size, mode='bilinear')
        out['depth'] = depth
        self.outputs[("disp", 0)] = depth
        return self.outputs