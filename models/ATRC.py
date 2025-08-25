import torch
import torch.nn as nn

#from . import utils_heads
from . import attention
from .base import BaseHead
import torch
import torch.nn as nn

import torchvision
import numpy as np
import math
import torch
import torch.nn as nn


class similarFunction(torch.autograd.Function):
    """ credit: https://github.com/zzd1992/Image-Local-Attention """

    @staticmethod
    def forward(ctx, x_ori, x_loc, kH, kW):
        ctx.save_for_backward(x_ori, x_loc)
        ctx.kHW = (kH, kW)
        output = locatt_ops.localattention.similar_forward(
            x_ori, x_loc, kH, kW)

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        x_ori, x_loc = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = locatt_ops.localattention.similar_backward(
            x_loc, grad_outputs, kH, kW, True)
        grad_loc = locatt_ops.localattention.similar_backward(
            x_ori, grad_outputs, kH, kW, False)

        return grad_ori, grad_loc, None, None


class weightingFunction(torch.autograd.Function):
    """ credit: https://github.com/zzd1992/Image-Local-Attention """

    @staticmethod
    def forward(ctx, x_ori, x_weight, kH, kW):
        ctx.save_for_backward(x_ori, x_weight)
        ctx.kHW = (kH, kW)
        output = locatt_ops.localattention.weighting_forward(
            x_ori, x_weight, kH, kW)

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        x_ori, x_weight = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = locatt_ops.localattention.weighting_backward_ori(
            x_weight, grad_outputs, kH, kW)
        grad_weight = locatt_ops.localattention.weighting_backward_weight(
            x_ori, grad_outputs, kH, kW)

        return grad_ori, grad_weight, None, None

class GlobalContextAttentionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, last_affine=True):
        super().__init__()
        self.eps = 1e-12
        self.query_project = nn.Sequential(ConvBNReLU(in_channels,
                                                                  out_channels,
                                                                  kernel_size=1,
                                                                  norm_layer=nn.BatchNorm2d,
                                                                  activation_layer=nn.ReLU),
                                           ConvBNReLU(out_channels,
                                                                  out_channels,
                                                                  kernel_size=1,
                                                                  norm_layer=nn.BatchNorm2d,
                                                                  activation_layer=None),
                                           nn.Softplus())
        self.key_project = nn.Sequential(ConvBNReLU(in_channels,
                                                                out_channels,
                                                                kernel_size=1,
                                                                norm_layer=nn.BatchNorm2d,
                                                                activation_layer=nn.ReLU),
                                         ConvBNReLU(out_channels,
                                                                out_channels,
                                                                kernel_size=1,
                                                                norm_layer=nn.BatchNorm2d,
                                                                activation_layer=None),
                                         nn.Softplus())
        self.value_project = ConvBNReLU(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    norm_layer=nn.BatchNorm2d,
                                                    activation_layer=nn.ReLU,
                                                    affine=last_affine)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, target_task_feats, source_task_feats, **kwargs):
        batch_size = target_task_feats.size(0)

        key = self.key_project(source_task_feats)  # b x d x h x w
        value = self.value_project(source_task_feats)  # b x m x h x w
        key = key.view(*key.shape[:2], -1)  # b x d x hw
        value = value.view(*value.shape[:2], -1)  # b x m x hw

        query = self.query_project(target_task_feats)  # b x d x h x w
        query = query.view(*query.shape[:2], -1)  # b x d x hw

        S = torch.matmul(value, key.permute(0, 2, 1))  # b x m x d
        Z = torch.sum(key, dim=2)  # b x d
        denom = torch.matmul(Z.unsqueeze(1), query)  # b x 1 x hw
        V = torch.matmul(S, query) / denom.clamp_min(self.eps)  # b x m x hw
        V = V.view(batch_size, -1, *target_task_feats.shape[2:]).contiguous()
        return V


class LocalContextAttentionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, last_affine=True):
        super().__init__()

        from .attention_ops import similarFunction, weightingFunction
        self.f_similar = similarFunction.apply
        self.f_weighting = weightingFunction.apply

        self.kernel_size = kernel_size
        self.query_project = nn.Sequential(ConvBNReLU(in_channels,
                                                                  out_channels,
                                                                  kernel_size=1,
                                                                  norm_layer=nn.BatchNorm2d,
                                                                  activation_layer=nn.ReLU),
                                           ConvBNReLU(out_channels,
                                                                  out_channels,
                                                                  kernel_size=1,
                                                                  norm_layer=nn.BatchNorm2d,
                                                                  activation_layer=nn.ReLU))
        self.key_project = nn.Sequential(ConvBNReLU(in_channels,
                                                                out_channels,
                                                                kernel_size=1,
                                                                norm_layer=nn.BatchNorm2d,
                                                                activation_layer=nn.ReLU),
                                         ConvBNReLU(out_channels,
                                                                out_channels,
                                                                kernel_size=1,
                                                                norm_layer=nn.BatchNorm2d,
                                                                activation_layer=nn.ReLU))
        self.value_project = ConvBNReLU(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    norm_layer=nn.BatchNorm2d,
                                                    activation_layer=nn.ReLU,
                                                    affine=last_affine)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, target_task_feats, source_task_feats, **kwargs):
        query = self.query_project(target_task_feats)
        key = self.key_project(source_task_feats)
        value = self.value_project(source_task_feats)

        weight = self.f_similar(query, key, self.kernel_size, self.kernel_size)
        weight = nn.functional.softmax(weight / math.sqrt(key.size(1)), -1)
        out = self.f_weighting(value, weight, self.kernel_size, self.kernel_size)
        return out


class LabelContextAttentionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, context_type, last_affine=True):
        super().__init__()
        self.context_type = context_type
        self.query_project = nn.Sequential(ConvBNReLU(in_channels,
                                                                  out_channels,
                                                                  kernel_size=1,
                                                                  norm_layer=nn.BatchNorm2d,
                                                                  activation_layer=nn.ReLU),
                                           ConvBNReLU(out_channels,
                                                                  out_channels,
                                                                  kernel_size=1,
                                                                  norm_layer=nn.BatchNorm2d,
                                                                  activation_layer=nn.ReLU))
        self.key_project = nn.Sequential(ConvBNReLU(in_channels,
                                                                out_channels,
                                                                kernel_size=1,
                                                                norm_layer=nn.BatchNorm2d,
                                                                activation_layer=nn.ReLU),
                                         ConvBNReLU(out_channels,
                                                                out_channels,
                                                                kernel_size=1,
                                                                norm_layer=nn.BatchNorm2d,
                                                                activation_layer=nn.ReLU))
        self.value_project = ConvBNReLU(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    norm_layer=nn.BatchNorm2d,
                                                    activation_layer=nn.ReLU,
                                                    affine=last_affine)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, target_task_feats, source_task_feats, target_aux_prob, source_aux_prob):
        context = self.gather_context(source_task_feats, target_aux_prob, source_aux_prob)
        batch_size = target_task_feats.size(0)

        key = self.key_project(context)
        value = self.value_project(context)
        key = key.view(*key.shape[:2], -1)
        value = value.view(*value.shape[:2], -1).permute(0, 2, 1)

        query = self.query_project(target_task_feats)
        query = query.view(*query.shape[:2], -1).permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map /= query.shape[-1]**0.5
        sim_map = sim_map.softmax(dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *target_task_feats.shape[2:])
        return context

    def gather_context(self, source_feats, target_aux_prob, source_aux_prob):
        if self.context_type == 'tlabel':
            batch_size, channels = source_feats.shape[:2]
            source_feats = source_feats.view(batch_size, channels, -1)
            source_feats = source_feats.permute(0, 2, 1)
            cxt = torch.matmul(target_aux_prob, source_feats)
            context = cxt.permute(0, 2, 1).contiguous().unsqueeze(3)
        elif self.context_type == 'slabel':
            batch_size, channels = source_feats.shape[:2]
            source_feats = source_feats.view(batch_size, channels, -1)
            source_feats = source_feats.permute(0, 2, 1)
            cxt = torch.matmul(source_aux_prob, source_feats)
            context = cxt.permute(0, 2, 1).contiguous().unsqueeze(3)
        return context
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, bias='auto',
                 inplace=True, affine=True):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.use_norm = norm_layer is not None
        self.use_activation = activation_layer is not None
        if bias == 'auto':
            bias = not self.use_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.use_norm:
            self.bn = norm_layer(out_channels, affine=affine)
        if self.use_activation:
            self.activation = activation_layer(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.bn(x)
        if self.use_activation:
            x = self.activation(x)
        return x


def spatial_normalize_pred(pred, image, ignore_index=255):
    prob = {}
    for t in pred.keys():
        task_pred = pred[t]
        batch_size, num_classes, H, W = task_pred.size()
        # check for ignore_index in input image, arising for example from data augmentation
        ignore_mask = (nn.functional.interpolate(image, size=(
            H, W), mode='nearest') == ignore_index).any(dim=1, keepdim=True)
        # so they won't contribute to the softmax
        task_pred[ignore_mask.expand_as(task_pred)] = -float('inf')
        c_probs = nn.functional.softmax(
            task_pred.view(batch_size, num_classes, -1), dim=2)
        # if the whole input image consisted of ignore regions, then context probs should just be zero
        prob[t] = torch.where(torch.isnan(
            c_probs), torch.zeros_like(c_probs), c_probs)
    return prob





def prep_a_net(model_name, shall_pretrain):
    model = getattr(torchvision.models, model_name)(shall_pretrain)
    if "resnet" in model_name:
        model.last_layer_name = 'fc'
    elif "mobilenet_v2" in model_name:
        model.last_layer_name = 'classifier'
    return model

def zero_pad(im, pad_size):
    """Performs zero padding (CHW format)."""
    pad_width = ((0, 0), (pad_size, pad_size), (pad_size, pad_size))
    return np.pad(im, pad_width, mode="constant")

def random_crop(im, size, pad_size=0):
    """Performs random crop (CHW format)."""
    if pad_size > 0:
        im = zero_pad(im=im, pad_size=pad_size)
    h, w = im.shape[1:]
    if size == h:
        return im
    y = np.random.randint(0, h - size)
    x = np.random.randint(0, w - size)
    im_crop = im[:, y : (y + size), x : (x + size)]
    assert im_crop.shape[1:] == (size, size)
    return im_crop

def get_patch(images, action_sequence, patch_size):
    """Get small patch of the original image"""
    batch_size = images.size(0)
    image_size = images.size(2)

    patch_coordinate = torch.floor(action_sequence * (image_size - patch_size)).int()
    patches = []
    for i in range(batch_size):
        per_patch = images[i, :,
                    (patch_coordinate[i, 0].item()): ((patch_coordinate[i, 0] + patch_size).item()),
                    (patch_coordinate[i, 1].item()): ((patch_coordinate[i, 1] + patch_size).item())]

        patches.append(per_patch.view(1, per_patch.size(0), per_patch.size(1), per_patch.size(2)))

    return torch.cat(patches, 0)

class ATRCModule(nn.Module):

    def __init__(self,
                 tasks,
                 atrc_genotype,
                 in_channels,
                 inter_channels,
                 drop_rate=0.05,
                 zero_init=True):
        super().__init__()
        self.tasks = tasks
        self.atrc_genotype = atrc_genotype

        self.cp_blocks = nn.ModuleDict()
        for target in self.tasks:
            self.cp_blocks[target] = nn.ModuleDict()
            for source in self.tasks:
                if atrc_genotype[target][source] == 0:  # none
                    pass
                elif atrc_genotype[target][source] == 1:  # global
                    self.cp_blocks[target][source] = attention.GlobalContextAttentionBlock(in_channels,
                                                                                           inter_channels)
                elif atrc_genotype[target][source] == 2:  # local
                    self.cp_blocks[target][source] = attention.LocalContextAttentionBlock(in_channels,
                                                                                          inter_channels,
                                                                                          kernel_size=9)
                elif atrc_genotype[target][source] == 3:  # t-label
                    self.cp_blocks[target][source] = attention.LabelContextAttentionBlock(in_channels,
                                                                                          inter_channels,
                                                                                          context_type='tlabel')
                elif atrc_genotype[target][source] == 4:  # s-label
                    self.cp_blocks[target][source] = attention.LabelContextAttentionBlock(in_channels,
                                                                                          inter_channels,
                                                                                          context_type='slabel')
                else:
                    raise ValueError

        self.out_proj = nn.ModuleDict()
        self.bottleneck = nn.ModuleDict()
        for target in self.tasks:
            nr_sources = len(list(self.cp_blocks[target].keys()))
            if nr_sources > 0:
                self.out_proj[target] = ConvBNReLU(inter_channels * nr_sources,
                                                               in_channels,
                                                               kernel_size=1,
                                                               norm_layer=nn.BatchNorm2d,
                                                               activation_layer=None)
                self.bottleneck[target] = nn.Sequential(ConvBNReLU(in_channels * 2,
                                                                               in_channels,
                                                                               kernel_size=1,
                                                                               norm_layer=nn.BatchNorm2d,
                                                                               activation_layer=nn.ReLU),
                                                        nn.Dropout2d(drop_rate))
            else:
                self.bottleneck[target] = nn.Sequential(ConvBNReLU(in_channels,
                                                                               in_channels,
                                                                               kernel_size=1,
                                                                               norm_layer=nn.BatchNorm2d,
                                                                               activation_layer=nn.ReLU),
                                                        nn.Dropout2d(drop_rate))
        if zero_init:
            for m in self.out_proj.values():
                if m.use_norm:
                    # initialize weight of last norm
                    nn.init.constant_(m.bn.weight, 0)
                    nn.init.constant_(m.bn.bias, 0)
                else:
                    # initialize weight of last conv
                    nn.init.constant_(m.conv.weight, 0)
                    nn.init.constant_(m.conv.bias, 0)

    def forward(self, task_specific_feats, aux_pred, image):
        aux_prob = spatial_normalize_pred(aux_pred, image)
        atrc_out_feats = {}
        for t in self.tasks:
            cp_out = []
            for s in self.tasks:
                if self.atrc_genotype[t][s] == 0:
                    continue
                cp_out.append(self.cp_blocks[t][s](target_task_feats=task_specific_feats[t],
                                                   source_task_feats=task_specific_feats[s],
                                                   target_aux_prob=aux_prob[t],
                                                   source_aux_prob=aux_prob[s]))
            if len(cp_out) > 0:
                distilled = torch.cat([task_specific_feats[t], self.out_proj[t](torch.cat(cp_out, dim=1))], dim=1)
            else:
                distilled = task_specific_feats[t]
            atrc_out_feats[t] = self.bottleneck[t](distilled)
        return atrc_out_feats


class RelationalContextHead(BaseHead):

    def __init__(self, atrc_genotype, **kwargs):
        super().__init__(**kwargs)
        self.atrc_genotype = atrc_genotype
        self.head_endpoints = ['final', 'aux']
        out_channels = self.in_channels // 4
        att_channels = out_channels // 2

        self.bottleneck = nn.ModuleDict({t: ConvBNReLU(self.in_channels,
                                                                   out_channels,
                                                                   kernel_size=3,
                                                                   norm_layer=nn.BatchNorm2d,
                                                                   activation_layer=nn.ReLU)
                                         for t in self.tasks})
        self.aux_tasks = self.get_aux_tasks()
        self.aux_conv = nn.ModuleDict({t: ConvBNReLU(self.in_channels,
                                                                 self.in_channels,
                                                                 kernel_size=1,
                                                                 norm_layer=nn.BatchNorm2d,
                                                                 activation_layer=nn.ReLU)
                                       for t in self.aux_tasks})
        self.aux_logits = nn.ModuleDict({t: nn.Conv2d(self.in_channels,
                                                      self.task_channel_mapping[t]['aux'],
                                                      kernel_size=1,
                                                      bias=True)
                                         for t in self.aux_tasks})
        self.atrc_module = ATRCModule(self.tasks,
                                      self.atrc_genotype,
                                      out_channels,
                                      att_channels)
        self.final_logits = nn.ModuleDict({t: nn.Conv2d(out_channels,
                                                        self.task_channel_mapping[t]['final'],
                                                        kernel_size=1,
                                                        bias=True)
                                           for t in self.tasks})
        self.init_weights()

    def forward(self, inp, inp_shape, image, **kwargs):
        inp = self._transform_inputs(inp)
        task_specific_feats = {t: self.bottleneck[t](inp) for t in self.tasks}

        aux_inp = inp.detach()  # no backprop from aux heads
        aux_pred = {t: self.aux_logits[t](self.aux_conv[t](aux_inp)) for t in self.aux_tasks}

        atrc_out_feats = self.atrc_module(task_specific_feats, aux_pred, image)
        final_pred = {t: self.final_logits[t](atrc_out_feats[t]) for t in self.tasks}

        final_pred = {t: nn.functional.interpolate(
            final_pred[t], size=inp_shape, mode='bilinear', align_corners=False) for t in self.tasks}
        aux_pred = {t: nn.functional.interpolate(
            aux_pred[t], size=inp_shape, mode='bilinear', align_corners=False) for t in self.aux_tasks}
        return {'final': final_pred, 'aux': aux_pred}

    def get_aux_tasks(self):
        # to make sure we only compute aux maps when necessary
        aux_tasks = []
        for task in self.tasks:
            if any(self.atrc_genotype[task][source] == 3 for source in self.tasks):
                aux_tasks.append(task)
                continue
            if any(self.atrc_genotype[target][task] == 4 for target in self.tasks):
                aux_tasks.append(task)
                continue
        return aux_tasks