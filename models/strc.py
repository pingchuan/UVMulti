import torch
import torch.nn as nn


import math

class BaseHead(nn.Module):
    def __init__(self, tasks, task_channel_mapping, in_index, idx_to_planes):
        super().__init__()
        self.tasks = tasks
        self.task_channel_mapping = task_channel_mapping
        self.in_index = in_index
        self.idx_to_planes = idx_to_planes
        self.in_channels = sum([self.idx_to_planes[i] for i in self.in_index])

    def forward(self, inp, inp_shape):
        raise NotImplementedError

    def _transform_inputs(self, inputs):
        inputs = [inputs[i] for i in self.in_index]
        upsampled_inputs = [
            nn.functional.interpolate(
                input=x,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=False) for x in inputs
        ]
        inputs = torch.cat(upsampled_inputs, dim=1)
        return inputs

    def init_weights(self):
        # By default we use pytorch default initialization. Heads can have their own init.
        # Except if `logits` is in the name, we override.
        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if 'logits' in name:
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

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


class similarFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x_ori, x_loc, kH, kW):
        """
        Forward pass of the similar function.

        :param x_ori: Original tensor
        :param x_loc: Location tensor
        :param kH: Kernel height
        :param kW: Kernel width
        """
        # Save inputs for backward computation
        ctx.save_for_backward(x_ori, x_loc)
        ctx.kHW = (kH, kW)

        # Assuming the forward operation is a form of convolution or local attention
        # Here, we'll use a simple convolution-like operation as an example
        output = F.conv2d(x_ori, x_loc, stride=1, padding=(kH // 2, kW // 2))

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        """
        Backward pass of the similar function.

        :param grad_outputs: Gradient of the output w.r.t some loss
        """
        x_ori, x_loc = ctx.saved_tensors
        kH, kW = ctx.kHW

        # Compute gradients for the original input and the location tensor
        grad_ori = F.conv_transpose2d(grad_outputs, x_loc, stride=1, padding=(kH // 2, kW // 2))
        grad_loc = F.conv_transpose2d(grad_outputs, x_ori, stride=1, padding=(kH // 2, kW // 2))

        return grad_ori, grad_loc, None, None


class weightingFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x_ori, x_weight, kH, kW):
        """
        Forward pass of the weighting function.

        :param x_ori: Original tensor
        :param x_weight: Weight tensor
        :param kH: Kernel height
        :param kW: Kernel width
        """
        # Save inputs for backward computation
        ctx.save_for_backward(x_ori, x_weight)
        ctx.kHW = (kH, kW)

        # Here, we're assuming the forward operation involves a weighted sum
        # We'll simply multiply the input by the weight tensor element-wise
        output = x_ori * x_weight

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        """
        Backward pass of the weighting function.

        :param grad_outputs: Gradient of the output w.r.t some loss
        """
        x_ori, x_weight = ctx.saved_tensors
        kH, kW = ctx.kHW

        # Compute gradients for the original input and the weight tensor
        grad_ori = grad_outputs * x_weight  # Gradient for original input
        grad_weight = grad_outputs * x_ori  # Gradient for weight tensor

        return grad_ori, grad_weight, None, None


class LocalContextAttentionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, last_affine=True):
        super().__init__()


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

class ConvexCombination(nn.Module):

    def __init__(self, op_list):
        super().__init__()
        self.op_list = op_list
        # also initialize weight for `none` op
        self.arch_param = nn.Parameter(torch.zeros(len(self.op_list) + 1))
        self.register_buffer('fixed_weights', None)

    def forward(self, gumbel_temp, mode='gumbel', **kwargs):
        if self.fixed_weights is None:
            if mode == 'gumbel':
                sort_arch_param = torch.topk(nn.functional.softmax(self.arch_param, dim=-1), 2)
                if sort_arch_param[0][0] - sort_arch_param[0][1] >= 0.3:
                    # end stochastic process irreversibly
                    self.fixed_weights = torch.zeros_like(
                        self.arch_param, requires_grad=False).scatter_(-1, sort_arch_param[1][0], 1.0)
                    weights = self.fixed_weights
                else:
                    weights = nn.functional.gumbel_softmax(self.arch_param, tau=gumbel_temp, hard=False, dim=-1)
            elif mode == 'argmax':
                index = self.arch_param.max(-1, keepdim=True)[1]
                weights = torch.zeros_like(self.arch_param, requires_grad=False).scatter_(-1, index, 1.0)
        else:
            weights = self.fixed_weights

        # weights[0] receives appropriate gradient through gumbel softmax
        out = sum(weights[i + 1] * op(**kwargs)for i, op in enumerate(self.op_list))
        return out


class ATRCSearchModule(nn.Module):

    def __init__(self,
                 tasks,
                 in_channels,
                 inter_channels,
                 zero_init=True):
        super().__init__()
        self.tasks = tasks

        self.cp_blocks = nn.ModuleDict()
        for target in self.tasks:
            self.cp_blocks[target] = nn.ModuleDict()
            for source in self.tasks:
                op_list = [
                    GlobalContextAttentionBlock(in_channels,
                                                          inter_channels,
                                                          last_affine=False),
                   LocalContextAttentionBlock(in_channels,
                                                         inter_channels,
                                                         kernel_size=9,
                                                         last_affine=False),
                    LabelContextAttentionBlock(in_channels,
                                                         inter_channels,
                                                         context_type='tlabel',
                                                         last_affine=False)
                ]
                if target != source:
                    op_list.append(LabelContextAttentionBlock(in_channels,
                                                                        inter_channels,
                                                                        context_type='slabel',
                                                                        last_affine=False))
                self.cp_blocks[target][source] = ConvexCombination(nn.ModuleList(op_list))

        self.out_proj = nn.ModuleDict({t: ConvBNReLU(inter_channels * len(self.tasks),
                                                                 in_channels,
                                                                 kernel_size=1,
                                                                 norm_layer=nn.BatchNorm2d,
                                                                 activation_layer=None)
                                       for t in self.tasks})
        self.bottleneck = nn.ModuleDict({t: ConvBNReLU(in_channels * 2,
                                                                   in_channels,
                                                                   kernel_size=1,
                                                                   norm_layer=nn.BatchNorm2d,
                                                                   activation_layer=nn.ReLU)
                                         for t in self.tasks})
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

    def forward(self, task_specific_feats, aux_pred, image, gumbel_temp, mode='gumbel'):
        aux_prob = spatial_normalize_pred(aux_pred, image)
        atrc_out_feats = {}
        for t in self.tasks:
            cp_out = []
            for s in self.tasks:
                cp_out.append(self.cp_blocks[t][s](target_task_feats=task_specific_feats[t],
                                                   source_task_feats=task_specific_feats[s],
                                                   target_aux_prob=aux_prob[t],
                                                   source_aux_prob=aux_prob[s],
                                                   gumbel_temp=gumbel_temp,
                                                   mode=mode))
            distilled = torch.cat([task_specific_feats[t], self.out_proj[t](torch.cat(cp_out, dim=1))], dim=1)
            atrc_out_feats[t] = self.bottleneck[t](distilled)
        return atrc_out_feats


class RelationalContextSearchHead(BaseHead):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.head_endpoints = ['final', 'aux']
        out_channels = self.in_channels // 4
        att_channels = out_channels // 2

        self.bottleneck = nn.ModuleDict({t: ConvBNReLU(self.in_channels,
                                                                   out_channels,
                                                                   kernel_size=3,
                                                                   norm_layer=nn.BatchNorm2d,
                                                                   activation_layer=nn.ReLU)
                                         for t in self.tasks})
        self.aux_conv = nn.ModuleDict({t: ConvBNReLU(self.in_channels,
                                                                 self.in_channels,
                                                                 kernel_size=1,
                                                                 norm_layer=nn.BatchNorm2d,
                                                                 activation_layer=nn.ReLU)
                                       for t in self.tasks})
        self.aux_logits = nn.ModuleDict({t: nn.Conv2d(self.in_channels,
                                                      self.task_channel_mapping[t]['aux'],
                                                      kernel_size=1,
                                                      bias=True)
                                        for t in self.tasks})
        self.atrc_module = ATRCSearchModule(self.tasks,
                                            out_channels,
                                            att_channels)
        self.final_logits = nn.ModuleDict({t: nn.Conv2d(out_channels,
                                                        self.task_channel_mapping[t]['final'],
                                                        kernel_size=1,
                                                        bias=True)
                                           for t in self.tasks})
        self.init_weights()

    def forward(self, inp, inp_shape, image, gumbel_temp, mode='gumbel'):
        self.outputs = {}
        inp = self._transform_inputs(inp)
        task_specific_feats = {t: self.bottleneck[t](inp) for t in self.tasks}

        aux_inp = inp.detach()  # no backprop from aux heads
        aux_pred = {t: self.aux_logits[t](self.aux_conv[t](aux_inp)) for t in self.tasks}

        atrc_out_feats = self.atrc_module(task_specific_feats, aux_pred, image, gumbel_temp=gumbel_temp, mode=mode)
        final_pred = {t: self.final_logits[t](atrc_out_feats[t]) for t in self.tasks}

        final_pred = {t: nn.functional.interpolate(
            final_pred[t], size=inp_shape, mode='bilinear', align_corners=False) for t in self.tasks}
        aux_pred = {t: nn.functional.interpolate(
            aux_pred[t], size=inp_shape, mode='bilinear', align_corners=False) for t in self.tasks}
        return {'final': final_pred, 'aux': aux_pred}