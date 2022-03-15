"""Create model."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2020-2022 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 09月 09日 星期三 23:56:45 CST
# ***
# ************************************************************************************/
#

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import functools
import numpy as np

from . import thops
from typing import List
from typing import Optional
from typing import Dict

import pdb


class LLFlow(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=32, nb=4, gc=32, scale=1, K=4):
        super(LLFlow, self).__init__()
        self.RRDB = ConEncoder1(in_nc, out_nc, nf, nb, gc, scale)
        hidden_channels = 64
        self.flowUpsamplerNet = FlowUpsamplerNet((160, 160, 3), hidden_channels, K)
        self.max_pool = nn.MaxPool2d(3)

    def forward(self, x):
        log_lr = torch.log(torch.clamp(x + 1e-3, min=1e-3))
        x255 = x * 255.0
        heq_lr = TF.equalize(x255.to(torch.uint8)).float() / 255.0
        lr = torch.cat((log_lr, 0.5*heq_lr.clamp(0, 1.0) + 0.5*x), dim=1)
        # lr.size()-- [1, 6, 400, 600]

        # make noise tensor
        eps_std = 0.1
        B, C, H, W = lr.shape
        size = (B, 3 * 8 * 8, H // 8, W // 8)
        z = torch.normal(mean=0.0, std=eps_std, size=size)
        # z.size() -- [1, 192, 50, 75]

        logdet = torch.zeros_like(lr[:, 0, 0, 0])
        
        rrdbResults = self.rrdbPreprocessing(lr)
        z = squeeze2d(rrdbResults["color_map"], 8)
        x, logdet = self.flowUpsamplerNet(rrdbResults, z, logdet, eps_std)

        # return x, logdet
        return x.clamp(0.0, 1.0)

    def rrdbPreprocessing(self, lr)->Dict[str, torch.Tensor]:
        rrdbResults = self.RRDB(lr)

        block_idxs = [1]  # opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'blocks']) or []
        low_level_features = [rrdbResults["block_{}".format(idx)] for idx in block_idxs]
        concat = torch.cat(low_level_features, dim=1)
        keys = ["last_lr_fea", "fea_up1", "fea_up2", "fea_up4"]
        if "fea_up0" in rrdbResults.keys():
            keys.append("fea_up0")
        if "fea_up-1" in rrdbResults.keys():
            keys.append("fea_up-1")
        for k in keys:
            h = rrdbResults[k].shape[2]
            w = rrdbResults[k].shape[3]
            rrdbResults[k] = torch.cat([rrdbResults[k], F.interpolate(concat, (h, w))], dim=1)

        return rrdbResults


class FlowUpsamplerNet(nn.Module):
    def __init__(self, image_shape, hidden_channels, K):

        super().__init__()

        self.hr_size = 160  # opt['datasets']['train']['GT_size']
        self.layer_list = [] # nn.ModuleList() # xxxx8888
        self.output_shapes: List[int] = []

        self.L = 3  # opt_get(opt, ['network_G', 'flow', 'L']) # 3
        self.K = 4  # opt_get(opt, ['network_G', 'flow', 'K']) # 4
        # if isinstance(self.K, int):
        #     self.K = [K for K in [K, ] * (self.L + 1)]
        self.K = [4, 4, 4, 4]
        H, W, self.C = image_shape

        self.levelToName = {
            # 0: 'fea_up4',
            1: "fea_up2",
            2: "fea_up1",
            3: "fea_up0",
            # 4: 'fea_up-1'
        }

        # Upsampler        
        # self.C, H, W -- (3, 160, 160)
        for level in range(1, self.L + 1):
            # 1. Squeeze
            H, W = self.arch_Squeeze(H, W)

            # 2. K FlowStep
            self.arch_FlowAffine(H, hidden_channels)
            self.arch_FlowStep(H, self.K[level], hidden_channels)
        # self.C, H, W -- (192, 20, 20)

        self.layers = nn.Sequential(*self.layer_list)

        self.f = f_conv2d_bias(128, 2 * 3 * 64) # 128 -- self.get_affineInCh(opt_get)


    def arch_FlowStep(self, H, K, hidden_channels):
        for k in range(K):
            self.layer_list.append(
                FlowStep(
                    in_channels=self.C,
                    hidden_channels=hidden_channels,
                    flow_permutation="invconv",
                    flow_coupling="CondAffineSeparatedAndCond",
                )
            )
            self.output_shapes.append(H)

    def arch_FlowAffine(self, H, hidden_channels):
        n_additionalFlowNoAffine = 2  # int(opt['network_G']['flow']['additionalFlowNoAffine'])
        for _ in range(n_additionalFlowNoAffine):
            self.layer_list.append(
                FlowStep(
                    in_channels=self.C,
                    hidden_channels=hidden_channels,
                    flow_permutation="invconv",
                    flow_coupling="noCoupling",
                )
            )

            self.output_shapes.append(H)

    def arch_Squeeze(self, H, W):
        self.C, H, W = self.C * 4, H // 2, W // 2
        self.layer_list.append(SqueezeLayer(factor=2))
        self.output_shapes.append(H)
        return H, W

    def forward(self, rrdbResults: Dict[str, torch.Tensor], z, logdet, eps_std: float):
        fl_fea = z

        level_conditionals: Dict[int, torch.Tensor] = {}
        for level in range(self.L + 1):
            if level not in self.levelToName.keys():
                level_conditionals[level] = torch.randn(1,3, 8, 8) # None, Fake for torchscript compile
            else:
                # level_conditionals[level] = rrdbResults[self.levelToName[level]] if rrdbResults else None
                level_conditionals[level] = rrdbResults[self.levelToName[level]]

        # for layer, size in zip(reversed(self.layers), reversed(self.output_shapes)):
        #     level = int(math.log(self.hr_size / size) / math.log(2))
        #     # FlowStep, SqueezeLayer
        #     if isinstance(layer, FlowStep):
        #         fl_fea, logdet = layer(fl_fea, logdet=logdet, rrdbResults=level_conditionals[level])
        #     else:
        #         fl_fea, logdet = layer(fl_fea, logdet=logdet)  # SqueezeLayer

        length = len(self.layers)
        for index in range(length):
            layer = self.layers[length - 1 - index]
            size = self.output_shapes[length - 1 - index]

            level = int(math.log(self.hr_size / size) / math.log(2))
            # FlowStep, SqueezeLayer
            if isinstance(layer, FlowStep):
                fl_fea, logdet = layer(fl_fea, logdet, rrdbResults=level_conditionals[level])
            elif isinstance(layer, SqueezeLayer):
                fl_fea, logdet = layer(fl_fea, logdet)  # SqueezeLayer
            else:
                pass

        sr = fl_fea
        # assert sr.shape[1] == 3
        return sr, logdet


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # ==> pdb.set_trace()
        # nf = 32
        # gc = 32
        # bias = True

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)
        # nf = 32
        # gc = 32

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class ConEncoder1(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4):
        super(ConEncoder1, self).__init__()

        in_nc = in_nc + 3  # concat_histeq
        in_nc = in_nc + 6
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.scale = scale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv_second = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### downsampling
        self.downconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.downconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.downconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.downconv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.awb_para = nn.Linear(nf, 3)
        self.fine_tune_color_map = nn.Sequential(nn.Conv2d(nf, 3, 1, 1), nn.Sigmoid())
        # in_nc = 12
        # out_nc = 3
        # nf = 32
        # nb = 4
        # gc = 32
        # scale = 1

    def forward(self, x):
        raw_low_input = x[:, 0:3].exp()
        awb_weight = 1  # (1 + self.awb_para(fea_for_awb).unsqueeze(2).unsqueeze(3))
        low_after_awb = raw_low_input * awb_weight
        color_map = low_after_awb / (low_after_awb.sum(dim=1, keepdim=True) + 1e-4)
        dx, dy = self.gradient(color_map)
        noise_map = torch.max(torch.stack([dx.abs(), dy.abs()], dim=0), dim=0)[0]

        fea = self.conv_first(torch.cat([x, color_map, noise_map], dim=1))
        fea = self.lrelu(fea)
        fea = self.conv_second(fea)
        fea_head = F.max_pool2d(fea, 2)

        block_idxs = [1]  # opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'blocks']) or []

        block_results = {}
        fea = fea_head
        for idx, m in enumerate(self.RRDB_trunk.children()):
            fea = m(fea)
            for b in block_idxs:
                if b == idx:
                    block_results["block_{}".format(idx)] = fea
        trunk = self.trunk_conv(fea)
        fea_down2 = fea_head + trunk
        fea_down4 = self.downconv1(F.interpolate(fea_down2, scale_factor=0.5, recompute_scale_factor=True))

        fea = self.lrelu(fea_down4)
        fea_down8 = self.downconv2(F.interpolate(fea, scale_factor=0.5, recompute_scale_factor=True))

        results = {
            "fea_up0": fea_down8,
            "fea_up1": fea_down4,
            "fea_up2": fea_down2,
            "fea_up4": fea_head,
            "last_lr_fea": fea_down4,
            "color_map": self.fine_tune_color_map(F.interpolate(fea_down2, scale_factor=2.0)),
        }

        for k, v in block_results.items():
            results[k] = v
        return results

    def sub_gradient(self, x):
        left_shift_x, right_shift_x, grad = torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)
        left_shift_x[:, :, 0:-1] = x[:, :, 1:]
        right_shift_x[:, :, 1:] = x[:, :, 0:-1]
        grad = 0.5 * (left_shift_x - right_shift_x)
        return grad

    def gradient(self, x) -> List[torch.Tensor]:
        return self.sub_gradient(x), self.sub_gradient(torch.transpose(x, 2, 3)).transpose(2, 3)


def squeeze2d(input, factor: int):
    if factor == 1:
        return input
    B, C, H, W = input.shape
    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)
    return x


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


def unsqueeze2d(input, factor: int):
    if factor == 1:
        return input
    B, C, H, W = input.shape
    x = input.view(B, C // (factor * factor), factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor * factor), H * factor, W * factor)
    return x


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet) -> List[torch.Tensor]:
        output = unsqueeze2d(input, self.factor)
        return output, logdet


class FlowStep(nn.Module):
    def __init__(self, in_channels, hidden_channels, flow_permutation="invconv", flow_coupling="additive"):
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling

        # 1. actnorm
        self.actnorm = ActNorm2d(in_channels)

        # 2. permute
        self.invconv = InvertibleConv1x1(in_channels)

        # 3. coupling
        self.need_features = False
        if flow_coupling == "CondAffineSeparatedAndCond":
            self.affine = CondAffineSeparatedAndCond(in_channels=in_channels)
            self.need_features = self.affine.need_features
        # elif flow_coupling == "noCoupling":
        #     self.affine = FakeAffineSeparatedAndCond(in_channels=in_channels)
        # else:
        #     raise RuntimeError("coupling not Found:", flow_coupling)
        else:
            self.affine = FakeAffineSeparatedAndCond(in_channels=in_channels)

    def forward(self, input, logdet, rrdbResults) -> List[torch.Tensor]:        
        return self.reverse_flow(input, logdet, rrdbResults)

    def reverse_flow(self, z, logdet, rrdbResults) -> List[torch.Tensor]:
        # 1.coupling
        if self.need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
            z, logdet = self.affine(z, logdet, rrdbResults)

        # 2. permute
        z, logdet = self.invconv(z, logdet)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet, reverse=True)

        return z, logdet


class ActNorm2d(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features):
        super().__init__()
        size = [1, num_features, 1, 1]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))

    def _center(self, input, reverse: bool = False):
        if not reverse:
            return input + self.bias
        else:
            return input - self.bias

    def _scale(self, input, logdet, reverse: bool = False) -> List[torch.Tensor]:
        '''logdet is not None'''

        logs = self.logs

        if not reverse:
            input = input * torch.exp(logs)
        else:
            input = input * torch.exp(-logs)

        """
        logs is log_std of `mean of channels`
        so we need to multiply pixels
        """
        dlogdet = torch.sum(logs) * thops.pixels(input)
        if reverse:
            dlogdet *= -1
        logdet = logdet + dlogdet
        return input, logdet

    def forward(self, input, logdet, reverse: bool = False) -> List[torch.Tensor]:
        if not reverse:
            # center and scale
            input = self._center(input, reverse)
            input, logdet = self._scale(input, logdet, reverse)
        else:
            input, logdet = self._scale(input, logdet, reverse)
            input = self._center(input, reverse)
        return input, logdet


    def less_scale(self, input, reverse: bool = False):
        '''logdet is None'''

        logs = self.logs

        if not reverse:
            input = input * torch.exp(logs)
        else:
            input = input * torch.exp(-logs)
        return input


    def less_forward(self, input, reverse: bool = False):
        '''logdet is None'''
        if not reverse:
            # center and scale
            input = self._center(input, reverse)
            input = self.less_scale(input, reverse)
        else:
            input = self.less_scale(input, reverse)
            input = self._center(input, reverse)
        return input


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.w_shape = w_shape
        # num_channels -- 12

    def get_weight(self, input) -> List[torch.Tensor]:
        w_shape = self.w_shape
        pixels = thops.pixels(input)
        dlogdet = torch.slogdet(self.weight)[1] * pixels
        weight = torch.inverse(self.weight.double()).float().view(w_shape[0], w_shape[1], 1, 1)

        return weight, dlogdet

    def forward(self, input, logdet) -> List[torch.Tensor]:
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input)
        z = F.conv2d(input, weight)
        if logdet is not None:
            logdet = logdet - dlogdet
        return z, logdet


class CondAffineSeparatedAndCond(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = 64  # opt_get(opt, ['network_G', 'flow', 'conditionInFeaDim'], 320) # 64
        self.hidden_channels = 64  # if hidden_channels is None else hidden_channels

        self.affine_eps = 0.0001  # opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'eps'], 0.0001)
        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn  # -- 6

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2
        # self.channels_for_nn -- 6

        self.fAffine = self.F(
            in_channels=self.channels_for_nn + self.in_channels_rrdb,
            out_channels=self.channels_for_co * 2,
            hidden_channels=self.hidden_channels,
            kernel_hidden=1,
            n_hidden_layers=1,
        )

        self.fFeatures = self.F(
            in_channels=self.in_channels_rrdb,
            out_channels=self.in_channels * 2,
            hidden_channels=self.hidden_channels,
            kernel_hidden=1,
            n_hidden_layers=1,
        )
        # in_channels = 12


    def forward(self, input, logdet, rrdbResults) -> List[torch.Tensor]:
        z = input

        # Self Conditional
        z1, z2 = self.split(z)
        scale, shift = self.feature_extract_aff(z1, rrdbResults)
        z2 = z2 / scale
        z2 = z2 - shift
        z = torch.cat((z1, z2), dim=1)
        logdet = logdet - self.get_logdet(scale)

        # Feature Conditional
        scaleFt, shiftFt = self.feature_extract(rrdbResults)
        z = z / scaleFt
        z = z - shiftFt
        logdet = logdet - self.get_logdet(scaleFt)

        output = z
        return output, logdet

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z) -> List[torch.Tensor]:
        # h = self.fFeatures(z)
        h = z
        for layer in self.fFeatures:
            if isinstance(layer, (Conv2d, Conv2dZeros)):
                h = layer.more_forward(h)
            else:
                h = layer(h)            

        shift, scale = thops.split_cross(h)
        scale = torch.sigmoid(scale + 2.0) + self.affine_eps
        return scale, shift

    def feature_extract_aff(self, z1, ft) -> List[torch.Tensor]:
        z = torch.cat([z1, ft], dim=1)

        # h = self.fAffine(z)
        h = z
        for layer in self.fAffine:
            if isinstance(layer, (Conv2d, Conv2dZeros)):
                h = layer.more_forward(h)
            else:
                h = layer(h)

        shift, scale = thops.split_cross(h)
        scale = torch.sigmoid(scale + 2.0) + self.affine_eps
        return scale, shift

    def split(self, z) -> List[torch.Tensor]:
        z1 = z[:, : self.channels_for_nn]
        z2 = z[:, self.channels_for_nn :]
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels=64, kernel_hidden=1, n_hidden_layers=1):
        # xxxx8888
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]
        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)


class Conv2d(nn.Conv2d):
    pad_dict = {
        "same": lambda kernel, stride: [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)],
        "valid": lambda kernel, stride: [0 for _ in kernel],
    }

    @staticmethod
    def get_padding(padding, kernel_size, stride):
        # make paddding
        if isinstance(padding, str):
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            if isinstance(stride, int):
                stride = [stride, stride]
            padding = padding.lower()
            try:
                padding = Conv2d.pad_dict[padding](kernel_size, stride)
            except KeyError:
                raise ValueError("{} is not supported".format(padding))
        return padding

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding="same",
        weight_std=0.05,
    ):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        # init weight with std
        self.weight.data.normal_(mean=0.0, std=weight_std)
        self.actnorm = ActNorm2d(out_channels)

    def more_forward(self, input):
        x = self.forward(input)
        x = self.actnorm.less_forward(x)  # less is more ...
        return x


class Conv2dZeros(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=[3, 3], stride=[1, 1], padding="same"):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.logscale_factor = 3.0
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        self.weight.data.zero_()
        self.bias.data.zero_()

    def more_forward(self, input):
        output = self.forward(input)
        return output * torch.exp(self.logs * self.logscale_factor) # more ...


def f_conv2d_bias(in_channels, out_channels):
    def padding_same(kernel, stride):
        return [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)]

    padding = padding_same([3, 3], [1, 1])
    assert padding == [1, 1], padding
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=[3, 3], stride=1, padding=1, bias=True
        )
    )


class FakeAffineSeparatedAndCond(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

    def forward(self, input, logdet, rrdbResults) -> List[torch.Tensor]:
        return input, logdet
