import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
#from models.resnet import BasicBlock
#from models.layers import SEBlock
#from models.padnet import MultiTaskDistillationModule
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

class MultiTaskDistillationModule(nn.Module):
    """
        Perform Multi-Task Distillation
        We apply an attention mask to features from other tasks and
        add the result as a residual.
    """

    def __init__(self, tasks, auxilary_tasks, channels):
        super(MultiTaskDistillationModule, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.self_attention = {}

        for t in self.tasks:
            other_tasks = [a for a in self.auxilary_tasks if a != t]
            self.self_attention[t] = nn.ModuleDict({a: SABlock(channels, channels) for a in other_tasks})
        self.self_attention = nn.ModuleDict(self.self_attention)

    def forward(self, x):
        adapters = {t: {a: self.self_attention[t][a](x['features_%s' % (a)]) for a in self.auxilary_tasks if a != t} for
                    t in self.tasks}
        out = {t: x['features_%s' % (t)] + torch.sum(torch.stack([v for v in adapters[t].values()]), dim=0) for t in
               self.tasks}
        return out
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
class InitialTaskPredictionModule(nn.Module):
    """ Module to make the inital task predictions """

    def __init__(self,  auxilary_tasks, input_channels, task_channels):
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


class MTINet(nn.Module):
    """
        MTI-Net implementation based on HRNet backbone
        https://arxiv.org/pdf/2001.06902.pdf
    """

    def __init__(self):
        super(MTINet, self).__init__()
        # General
        #self.tasks = p.TASKS.NAMES
        self.tasks = ['seg', 'enh', 'depth']
        #self.auxilary_tasks = p.AUXILARY_TASKS.NAMES
        self.auxilary_tasks = ['seg', 'enh', 'depth']

        self.channels = [64,128,256,512]
        self.num_scales = len(self.channels)
        # Backbone


        # Feature Propagation Module
        self.fpm_scale_3 = FPM(self.auxilary_tasks, self.channels[3])
        self.fpm_scale_2 = FPM(self.auxilary_tasks, self.channels[2])
        self.fpm_scale_1 = FPM(self.auxilary_tasks, self.channels[1])

        # Initial task predictions at multiple scales
        self.scale_0 = InitialTaskPredictionModule( self.auxilary_tasks, self.channels[0] + self.channels[1],
                                                   self.channels[0])
        self.scale_1 = InitialTaskPredictionModule( self.auxilary_tasks, self.channels[1] + self.channels[2],
                                                   self.channels[1])
        self.scale_2 = InitialTaskPredictionModule( self.auxilary_tasks, self.channels[2] + self.channels[3],
                                                   self.channels[2])
        self.scale_3 = InitialTaskPredictionModule(self.auxilary_tasks, self.channels[3], self.channels[3])

        # Distillation at multiple scales
        self.distillation_scale_0 = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, self.channels[0])
        self.distillation_scale_1 = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, self.channels[1])
        self.distillation_scale_2 = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, self.channels[2])
        self.distillation_scale_3 = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, self.channels[3])
        self.heads={}
        # Feature aggregation through Deeplab heads
        #self.heads["seg"] = HighResolutionHead(backbone_channels=[64,128,256,512],num_outputs=6)
        #self.heads["enh"] = HighResolutionHead(backbone_channels=[64,128,256,512],num_outputs=3)
        #self.heads["depth"] = HighResolutionHead(backbone_channels=[64,128,256,512],num_outputs=1)
        # Get device (GPU if available)
        self.heads["seg"] = Decoder1(num_classes=6)
        self.heads["enh"] = Decoder1(num_classes=3)
        self.heads["depth"] = Decoder1(num_classes=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move all heads to the correct device
        for key in self.heads:
            self.heads[key].to(self.device)
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

        # Distillation + Output
        features_0 = self.distillation_scale_0(x_0)
        features_1 = self.distillation_scale_1(x_1)
        features_2 = self.distillation_scale_2(x_2)
        features_3 = self.distillation_scale_3(x_3)
        multi_scale_features = {t: [features_0[t], features_1[t], features_2[t], features_3[t]] for t in self.tasks}

        # Feature aggregation
        for t in self.tasks:
            #out[t] = F.interpolate(self.heads[t](multi_scale_features[t]),[320, 320], mode='bilinear')
            self.outputs[t] = F.interpolate(self.heads[t](multi_scale_features[t]), [320, 320], mode='bilinear')


        # 对应的任务名称（'seg', 'enh', 'depth'）已经加入了 self.outputs 字典

        # 为了让你可以正确输出，需要用下面的方式访问：
        # self.outputs['seg'], self.outputs['enh'], self.outputs['depth']

        # 若你想要把 'depth' 输出改为 ('disp', 0) 也可以这样做：
        self.outputs[('disp', 0)] = torch.sigmoid(self.outputs['depth'])
        return self.outputs
class DeepLabHead1(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead1, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )
class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super(DecoderBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = ConvBlock(in_channels // 4, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x

class Decoder1(nn.Module):
    def __init__(self, num_classes):
        super(Decoder1, self).__init__()


        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        #self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)
        self.conv1 = nn.Conv2d(64, num_classes,
                              kernel_size=3,
                              stride=1,
                              padding=1)



    def forward(self, feature):
        e2, e3, e4, e5 = feature
        d5 = self.decoder5(e5)  # 14
        d4 = self.decoder4(torch.cat((d5, e4), dim=1))  # 28
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))  # 56
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))  # 128
        #d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # 224*224*64
        d2 =self.conv1(d2)

        return d2
class HighResolutionHead(nn.Module):
    def __init__(self, backbone_channels, num_outputs):
        super(HighResolutionHead, self).__init__()
        last_inp_channels = sum(backbone_channels)
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=0.1),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=num_outputs,
                kernel_size=1,
                stride=1,
                padding=0))

    def forward(self, x):
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], (x0_h, x0_w), mode='bilinear')
        x2 = F.interpolate(x[2], (x0_h, x0_w), mode='bilinear')
        x3 = F.interpolate(x[3], (x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.last_layer(x)
        return x
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)