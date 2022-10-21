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

from typing import List
from typing import Dict

import pdb


def thops_sum(tensor, dim: List[int]):
    dim = sorted(dim)
    for d in dim:
        tensor = tensor.sum(dim=d, keepdim=True)
    for i, d in enumerate(dim):
        tensor.squeeze_(d - i)
    return tensor


def thops_split_cross(tensor) -> List[torch.Tensor]:
    return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


class LLFlow(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=32, nb=4, gc=32):
        super(LLFlow, self).__init__()
        self.RRDB = ConEncoder1(in_nc, out_nc, nf, nb, gc)
        hidden_channels = 64
        self.flowUpsamplerNet = FlowUpsamplerNet((160, 160, 3), hidden_channels)
        # self.max_pool = nn.MaxPool2d(3)

    def forward_x(self, x):
        log_lr = torch.log(torch.clamp(x + 1e-3, min=1e-3))
        x255 = x * 255.0
        heq_lr = TF.equalize(x255.to(torch.uint8)).float() / 255.0
        lr = torch.cat((log_lr, 0.5 * heq_lr.clamp(0, 1.0) + 0.5 * x), dim=1)
        # lr.size()-- [1, 6, 400, 600]

        logdet = torch.zeros_like(lr[:, 0, 0, 0])
        # type(logdet) -- <class 'torch.Tensor'>, logdet.size() -- torch.Size([1])

        rrdbResults = self.rrdbPreprocessing(lr)
        color_map = squeeze2d(rrdbResults["color_map"], 8)
        y = self.flowUpsamplerNet(rrdbResults, color_map, logdet)

        return y.clamp(0.0, 1.0)

    def forward(self, x):
        # Define max GPU/CPU memory -- 2G
        max_h = 1024
        max_W = 1024
        multi_times = 8

        # Need Resize ?
        B, C, H, W = x.size()
        if H > max_h or W > max_W:
            s = min(max_h / H, max_W / W)
            SH, SW = int(s * H), int(s * W)
            resize_x = F.interpolate(x, size=(SH, SW), mode="bilinear", align_corners=False)
        else:
            resize_x = x

        # Need Zero Pad ?
        ZH, ZW = resize_x.size(2), resize_x.size(3)
        if ZH % multi_times != 0 or ZW % multi_times != 0:
            NH = multi_times * math.ceil(ZH / multi_times)
            NW = multi_times * math.ceil(ZW / multi_times)
            resize_zeropad_x = resize_x.new_zeros(B, C, NH, NW)
            resize_zeropad_x[:, :, 0:ZH, 0:ZW] = resize_x
        else:
            resize_zeropad_x = resize_x

        # MS Begin
        y = self.forward_x(resize_zeropad_x)
        del resize_zeropad_x, resize_x  # Release memory !!!

        y = y[:, :, 0:ZH, 0:ZW]  # Remove Zero Pads
        if ZH != H or ZW != W:
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        # MS End

        return y

    def rrdbPreprocessing(self, lr) -> Dict[str, torch.Tensor]:
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
    def __init__(self, image_shape, hidden_channels):
        super(FlowUpsamplerNet, self).__init__()

        self.hr_size = 160  # opt['datasets']['train']['GT_size']
        self.L = 3  # opt_get(opt, ['network_G', 'flow', 'L']) # 3
        self.K = [4, 4, 4, 4]
        H, W, self.C = image_shape

        self.levelToName = {
            # 0: 'fea_up4',
            1: "fea_up2",
            2: "fea_up1",
            3: "fea_up0",
            # 4: 'fea_up-1'
        }

        # self.C, H, W -- (3, 160, 160)
        layer_list = []
        self.output_shapes: List[int] = []
        for level in range(1, self.L + 1):
            # init_Squeeze
            self.C, H, W = self.C * 4, H // 2, W // 2
            layer_list.append(SqueezeLayer(factor=2))
            self.output_shapes.append(H)

            # init_FlowAffine
            # 2 -- int(opt['network_G']['flow']['additionalFlowNoAffine'])
            for _ in range(2):
                layer_list.append(
                    FlowStep(
                        in_channels=self.C,
                        hidden_channels=hidden_channels,
                        flow_coupling="noCoupling",
                    )
                )
                self.output_shapes.append(H)
            # init_FlowStep
            for k in range(self.K[level]):
                layer_list.append(
                    FlowStep(
                        in_channels=self.C,
                        hidden_channels=hidden_channels,
                        flow_coupling="CondAffineSeparatedAndCond",
                    )
                )
                self.output_shapes.append(H)

        # self.C, H, W -- (192, 20, 20)
        self.layers = nn.Sequential(*reversed([l for l in layer_list]))
        # self.f = f_conv2d_bias(128, 2 * 3 * 64)  # 128 -- self.get_affineInCh(opt_get)

    def forward(self, rrdbResults: Dict[str, torch.Tensor], color_map, logdet):
        level_conditionals: Dict[int, torch.Tensor] = {}
        for level in range(self.L + 1):
            if level not in self.levelToName.keys():
                level_conditionals[level] = torch.randn(1, 3, 8, 8)  # None, Fake for torchscript compile
            else:
                # level_conditionals[level] = rrdbResults[self.levelToName[level]] if rrdbResults else None
                level_conditionals[level] = rrdbResults[self.levelToName[level]]

        length = len(self.layers)
        for index, layer in enumerate(self.layers):
            size = self.output_shapes[length - 1 - index]
            level = int(math.log(self.hr_size / size) / math.log(2))

            x: List[torch.Tensor] = [color_map, logdet, level_conditionals[level]]
            y = layer(x)  # SqueezeLayer, FlowStep
            color_map, logdet = y[0], y[1]
        return color_map


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
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(ConEncoder1, self).__init__()

        in_nc = in_nc + 3  # concat histeq ==> 6
        in_nc = in_nc + 6  # Add flow channels ==> 12
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv_second = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        #### downsampling
        self.downconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.downconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.downconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

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

    def forward(self, x) -> Dict[str, torch.Tensor]:
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
        super(SqueezeLayer, self).__init__()
        self.factor = factor

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        input, logdet, rrdb = x[0], x[1], x[2]
        output = unsqueeze2d(input, self.factor)
        return output, logdet


class FlowStep(nn.Module):
    def __init__(self, in_channels, hidden_channels, flow_coupling):
        super(FlowStep, self).__init__()
        self.flow_coupling = flow_coupling

        # 1. actnorm
        self.actnorm = ActNorm2d(in_channels)

        # 2. permute
        self.invconv = InvertibleConv1x1(in_channels)

        # 3. coupling
        if flow_coupling == "CondAffineSeparatedAndCond":
            self.affine = CondAffineSeparatedAndCond(in_channels=in_channels)
        else:
            self.affine = FakeAffineSeparatedAndCond(in_channels=in_channels)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        input, logdet, rrdb = x[0], x[1], x[2]

        # 1.coupling
        input, logdet = self.affine([input, logdet, rrdb])

        # 2. permute
        input, logdet = self.invconv([input, logdet])

        # 3. actnorm
        input, logdet = self.actnorm(input, logdet, True)  # reverse=True

        return input, logdet


class ActNorm2d(nn.Module):
    """
    Activation Normalization
    """

    def __init__(self, num_features):
        super(ActNorm2d, self).__init__()
        size = [1, num_features, 1, 1]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))

    def _center(self, input, reverse: bool = False):
        if not reverse:
            return input + self.bias
        else:
            return input - self.bias

    def _scale(self, input, logdet, reverse: bool = False) -> List[torch.Tensor]:
        """logdet is not None"""

        logs = self.logs

        if not reverse:
            input = input * torch.exp(logs)
        else:
            input = input * torch.exp(-logs)

        """
        logs is log_std of `mean of channels`
        so we need to multiply pixels
        """
        dlogdet = torch.sum(logs) * (input.size(2) * input.size(3))
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
        """logdet is None"""

        logs = self.logs

        if not reverse:
            input = input * torch.exp(logs)
        else:
            input = input * torch.exp(-logs)
        return input

    def less_forward(self, input, reverse: bool):
        """logdet is None"""

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
        super(InvertibleConv1x1, self).__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.w_shape = w_shape
        # num_channels -- 12

    def get_weight(self, input) -> List[torch.Tensor]:
        w_shape = self.w_shape
        pixels = input.size(2) * input.size(3)
        dlogdet = torch.slogdet(self.weight)[1] * float(pixels)
        weight = torch.inverse(self.weight.double()).float().view(w_shape[0], w_shape[1], 1, 1)

        return weight, dlogdet

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        input, logdet = x[0], x[1]
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input)
        z = F.conv2d(input, weight)
        # if logdet is not None:
        logdet = logdet - dlogdet

        return z, logdet


class CondAffineSeparatedAndCond(nn.Module):
    def __init__(self, in_channels):
        super(CondAffineSeparatedAndCond, self).__init__()
        # self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = 64  # opt_get(opt, ['network_G', 'flow', 'conditionInFeaDim'], 320) # 64
        self.hidden_channels = 64  # if hidden_channels is None else hidden_channels

        self.affine_eps = 0.0001  # opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'eps'], 0.0001)
        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn  # -- 6

        # if self.channels_for_nn is None:
        #     self.channels_for_nn = self.in_channels // 2
        # self.channels_for_nn -- 6

        # self.fAffine = self.F(
        #     in_channels=self.channels_for_nn + self.in_channels_rrdb,
        #     out_channels=self.channels_for_co * 2,
        #     hidden_channels=self.hidden_channels,
        # )

        in_channels = self.channels_for_nn + self.in_channels_rrdb
        out_channels = self.channels_for_co * 2
        hidden_channels = self.hidden_channels

        layers = [
            Conv2dOnes(in_channels, hidden_channels, kernel_size=[3, 3], stride=[1, 1]),
            nn.ReLU(inplace=False),
            Conv2dOnes(hidden_channels, hidden_channels, kernel_size=[1, 1]),
            nn.ReLU(inplace=False),
            Conv2dZeros(hidden_channels, out_channels),
        ]
        self.fAffine = nn.Sequential(*layers)

        # self.fFeatures = self.F(
        #     in_channels=self.in_channels_rrdb,
        #     out_channels=self.in_channels * 2,
        #     hidden_channels=self.hidden_channels,
        # )

        in_channels = self.in_channels_rrdb
        out_channels = self.in_channels * 2
        hidden_channels = self.hidden_channels
        layers = [
            Conv2dOnes(in_channels, hidden_channels, kernel_size=[3, 3], stride=[1, 1]),
            nn.ReLU(inplace=False),
            Conv2dOnes(hidden_channels, hidden_channels, kernel_size=[1, 1]),
            nn.ReLU(inplace=False),
            Conv2dZeros(hidden_channels, out_channels),
        ]
        self.fFeatures = nn.Sequential(*layers)

        # in_channels = 12

    # def F(self, in_channels, out_channels, hidden_channels=64):
    #     layers = [
    #         Conv2dOnes(in_channels, hidden_channels, kernel_size=[3, 3], stride=[1, 1]),
    #         nn.ReLU(inplace=False),
    #         Conv2dOnes(hidden_channels, hidden_channels, kernel_size=[1, 1]),
    #         nn.ReLU(inplace=False),
    #         Conv2dZeros(hidden_channels, out_channels)
    #     ]
    #     return nn.Sequential(*layers)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        z, logdet, rrdb = x[0], x[1], x[2]

        # Self Conditional
        z1, z2 = self.split(z)
        scale, shift = self.feature_extract_aff([z1, rrdb])

        z2 = z2 / scale
        z2 = z2 - shift
        z = torch.cat((z1, z2), dim=1)
        logdet = logdet - self.get_logdet(scale)

        # Feature Conditional
        scaleFt, shiftFt = self.feature_extract(rrdb)
        z = z / scaleFt
        z = z - shiftFt
        logdet = logdet - self.get_logdet(scaleFt)

        return z, logdet

    def get_logdet(self, scale):
        return thops_sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z) -> List[torch.Tensor]:
        h = self.fFeatures(z)

        shift, scale = thops_split_cross(h)
        scale = torch.sigmoid(scale + 2.0) + self.affine_eps

        return scale, shift

    def feature_extract_aff(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        z1, ft = x[0], x[1]

        z = torch.cat([z1, ft], dim=1)
        h = self.fAffine(z)
        shift, scale = thops_split_cross(h)
        scale = torch.sigmoid(scale + 2.0) + self.affine_eps

        return scale, shift

    def split(self, z) -> List[torch.Tensor]:
        z1 = z[:, : self.channels_for_nn]
        z2 = z[:, self.channels_for_nn :]
        return z1, z2


def get_padding(padding, kernel_size, stride):
    pad_dict = {
        "same": lambda kernel, stride: [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)],
        "valid": lambda kernel, stride: [0 for _ in kernel],
    }

    # make paddding
    if isinstance(padding, str):
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        if isinstance(stride, int):
            stride = [stride, stride]
        padding = padding.lower()
        try:
            padding = pad_dict[padding](kernel_size, stride)
        except KeyError:
            raise ValueError("{} is not supported".format(padding))
    return padding


class Conv2dOnes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[3, 3], stride=[1, 1], padding_mode="same"):
        super(Conv2dOnes, self).__init__()

        padding = get_padding(padding_mode, kernel_size, stride)
        self.stdconv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

        # init weight with std
        weight_std = 0.05
        self.stdconv.weight.data.normal_(mean=0.0, std=weight_std)
        self.actnorm = ActNorm2d(out_channels)

    def forward(self, input):
        x = self.stdconv(input)
        return self.actnorm.less_forward(x, False)  # less is more ...


class Conv2dZeros(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[3, 3], stride=[1, 1], padding_mode="same"):
        super(Conv2dZeros, self).__init__()

        padding = get_padding(padding_mode, kernel_size, stride)
        self.stdconv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.logscale_factor = 3.0
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        self.stdconv.weight.data.zero_()
        self.stdconv.bias.data.zero_()

    def forward(self, input):
        output = self.stdconv(input)
        return output * torch.exp(self.logs * self.logscale_factor)  # more ...


class FakeAffineSeparatedAndCond(nn.Module):
    def __init__(self, in_channels):
        super(FakeAffineSeparatedAndCond, self).__init__()

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        input, logdet, rrdb = x[0], x[1], x[2]
        return input, logdet
