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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import functools
import numpy as np
from . import thops
from typing import List

import pdb


class LLFlow(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=32, nb=4, gc=32, scale=1, K=4):
        super(LLFlow, self).__init__()
        self.RRDB = ConEncoder1(in_nc, out_nc, nf, nb, gc, scale)
        hidden_channels = 64
        self.flowUpsamplerNet = FlowUpsamplerNet(
            (160, 160, 3), hidden_channels, K, flow_coupling="CondAffineSeparatedAndCond"
        )
        self.max_pool = nn.MaxPool2d(3)

    def forward(self, x):
        log_lr = torch.log(torch.clamp(x + 1e-3, min=1e-3))
        x255 = x * 255.0
        heq_lr = TF.equalize(x255.to(torch.uint8)).float()/255.0
        lr = torch.cat((log_lr, heq_lr), dim = 1)
        # lr.size()-- [1, 6, 400, 600]

        # make noise tensor
        eps_std = 0.5
        B, C, H, W = lr.shape
        size = (B, 3 * 8 * 8, H // 8, W // 8)
        z = torch.normal(mean=0, std=eps_std, size=size)
        # z.size() -- [1, 192, 50, 75]

        logdet = torch.zeros_like(lr[:, 0, 0, 0])
        lr_enc = self.rrdbPreprocessing(lr)
        z = squeeze2d(lr_enc["color_map"], 8)
        x, logdet = self.flowUpsamplerNet(rrdbResults=lr_enc, z=z, eps_std=eps_std, logdet=logdet)

        # return x, logdet
        return x.clamp(0.0, 1.0)

    def rrdbPreprocessing(self, lr):
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
    def __init__(
        self, image_shape, hidden_channels, K, L=None, actnorm_scale=1.0, flow_permutation=None, flow_coupling="affine"
    ):

        super().__init__()

        self.hr_size = 160  # opt['datasets']['train']['GT_size']
        self.layers = nn.ModuleList()
        self.output_shapes = []

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

        affineInCh = 128  # self.get_affineInCh(opt_get) # 128
        flow_permutation = "invconv"  # self.get_flow_permutation(flow_permutation, opt) # 'invconv'

        # Upsampler
        for level in range(1, self.L + 1):
            # 1. Squeeze
            H, W = self.arch_squeeze(H, W)

            # 2. K FlowStep
            self.arch_additionalFlowAffine(H, W, actnorm_scale, hidden_channels)
            self.arch_FlowStep(
                H, self.K[level], W, actnorm_scale, affineInCh, flow_coupling, flow_permutation, hidden_channels
            )

        self.f = f_conv2d_bias(affineInCh, 2 * 3 * 64)

    def arch_FlowStep(self, H, K, W, actnorm_scale, affineInCh, flow_coupling, flow_permutation, hidden_channels):

        for k in range(K):
            self.layers.append(
                FlowStep(
                    in_channels=self.C,
                    hidden_channels=hidden_channels,
                    actnorm_scale=actnorm_scale,
                    flow_permutation=flow_permutation,
                    flow_coupling=flow_coupling,
                )
            )
            self.output_shapes.append([-1, self.C, H, W])

    def arch_additionalFlowAffine(self, H, W, actnorm_scale, hidden_channels):
        n_additionalFlowNoAffine = 2  # int(opt['network_G']['flow']['additionalFlowNoAffine'])
        for _ in range(n_additionalFlowNoAffine):
            self.layers.append(
                FlowStep(
                    in_channels=self.C,
                    hidden_channels=hidden_channels,
                    actnorm_scale=actnorm_scale,
                    flow_permutation="invconv",
                    flow_coupling="noCoupling",
                )
            )
            self.output_shapes.append([-1, self.C, H, W])

    def arch_squeeze(self, H, W):
        self.C, H, W = self.C * 4, H // 2, W // 2
        self.layers.append(SqueezeLayer(factor=2))
        self.output_shapes.append([-1, self.C, H, W])
        return H, W

    def forward(self, rrdbResults=None, z=None, logdet=0.0, eps_std=None):
        sr, logdet = self.decode(rrdbResults, z, eps_std, logdet=logdet)
        return sr, logdet

    def decode(self, rrdbResults, z, eps_std=None, logdet=0.0):
        fl_fea = z
        level_conditionals = {}
        for level in range(self.L + 1):
            if level not in self.levelToName.keys():
                level_conditionals[level] = None
            else:
                level_conditionals[level] = rrdbResults[self.levelToName[level]] if rrdbResults else None

        for layer, shape in zip(reversed(self.layers), reversed(self.output_shapes)):
            size = shape[2]
            level = int(np.log(self.hr_size / size) / np.log(2))
            # FlowStep.FlowStep, flow.SqueezeLayer
            if isinstance(layer, FlowStep):
                fl_fea, logdet = layer(fl_fea, logdet=logdet, rrdbResults=level_conditionals[level])
            else:
                fl_fea, logdet = layer(fl_fea, logdet=logdet, reverse=True)

        sr = fl_fea
        assert sr.shape[1] == 3
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

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
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

        fea_down4 = self.downconv1(
            F.interpolate(
                fea_down2, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=True
            )
        )
        fea = self.lrelu(fea_down4)

        fea_down8 = self.downconv2(
            F.interpolate(fea, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=True)
        )

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
    # assert factor >= 1 and isinstance(factor, int)
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    # assert H % factor == 0 and W % factor == 0, "{}".format((H, W, factor))
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
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    x = input.view(B, C // (factor * factor), factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor * factor), H * factor, W * factor)
    return x


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet, reverse:bool=False) -> List[torch.Tensor]:
        if not reverse:
            output = squeeze2d(input, self.factor)  # Squeeze in forward
            return output, logdet
        else:
            output = unsqueeze2d(input, self.factor)
            return output, logdet


class FlowStep(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, actnorm_scale=1.0, flow_permutation="invconv", flow_coupling="additive"
    ):
        # check configures
        # assert flow_permutation in FlowStep.FlowPermutation, \
        #     "float_permutation should be in `{}`".format(FlowStep.FlowPermutation.keys())
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling


        # 1. actnorm
        self.actnorm = ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        self.invconv = InvertibleConv1x1(in_channels)

        # 3. coupling
        self.need_features = False
        if flow_coupling == "CondAffineSeparatedAndCond":
            self.affine = CondAffineSeparatedAndCond(in_channels=in_channels)
            self.need_features = self.affine.need_features
        elif flow_coupling == "noCoupling":
            pass
        else:
            raise RuntimeError("coupling not Found:", flow_coupling)

    def forward(self, input, logdet, rrdbResults: List[torch.Tensor])-> List[torch.Tensor]:
        return self.reverse_flow(input, logdet, rrdbResults)

    def reverse_flow(self, z, logdet, rrdbResults:bool=None):

        # need_features = self.affine_need_features()  # True
        # print("need_features: ", need_features)

        # 1.coupling
        # need_features:  True self.flow_coupling:  CondAffineSeparatedAndCond
        # need_features:  False self.flow_coupling:  noCoupling
        if self.need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
            z, logdet = self.affine(z, logdet, rrdbResults, True)

        # 2. permute
        z, logdet = self.invconv(z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet

    # def affine_need_features(self):
    #     need_features = False
    #     try:
    #         need_features = self.affine.need_features
    #     except:
    #         pass
    #     return need_features


class _ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.0):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))
        self.num_features = num_features
        self.scale = float(scale)
        self.inited = False

    def _check_input_dim(self, input):
        return NotImplemented

    def initialize_parameters(self, input):
        self._check_input_dim(input)
        if not self.training:
            return
        if (self.bias != 0).any():
            self.inited = True
            return
        assert input.device == self.bias.device, (input.device, self.bias.device)
        with torch.no_grad():
            bias = thops.mean(input.clone(), dim=[0, 2, 3]) * -1.0
            vars = thops.mean((input.clone() + bias) ** 2, dim=[0, 2, 3])
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.inited = True

    def _center(self, input, offset=None, reverse=False):
        bias = self.bias

        if offset is not None:
            bias = bias + offset

        if not reverse:
            return input + bias
        else:
            return input - bias

    def _scale(self, input, logdet=None, offset=None, reverse=False):
        logs = self.logs

        if offset is not None:
            logs = logs + offset

        if not reverse:
            input = input * torch.exp(logs)  # should have shape batchsize, n_channels, 1, 1
            # input = input * torch.exp(logs+logs_offset)
        else:
            input = input * torch.exp(-logs)
        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply pixels
            """
            dlogdet = torch.sum(logs) * thops.pixels(input)
            if reverse:
                dlogdet *= -1
            logdet = logdet + dlogdet
        return input, logdet

    def forward(self, input, logdet=None, offset_mask=None, logs_offset=None, bias_offset=None, reverse=False):
        if not self.inited:
            self.initialize_parameters(input)
        self._check_input_dim(input)

        if offset_mask is not None:
            logs_offset *= offset_mask
            bias_offset *= offset_mask
        # no need to permute dims as old version
        if not reverse:
            # center and scale

            # self.input = input
            input = self._center(input, bias_offset, reverse)
            input, logdet = self._scale(input, logdet, logs_offset, reverse)
        else:
            # scale and center
            input, logdet = self._scale(input, logdet, logs_offset, reverse)
            input = self._center(input, bias_offset, reverse)
        return input, logdet


class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.0):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 4
        assert (
            input.size(1) == self.num_features
        ), "[ActNorm]: input should be in shape as `BCHW`," " channels should be {} rather than {}".format(
            self.num_features, input.size()
        )


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.w_shape = w_shape
        # num_channels -- 12

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        pixels = thops.pixels(input)
        dlogdet = torch.tensor(float("inf"))
        while torch.isinf(dlogdet):
            try:
                dlogdet = torch.slogdet(self.weight)[1] * pixels
            except Exception as e:
                print(e)
                dlogdet = (
                    torch.slogdet(
                        self.weight + (self.weight.mean() * torch.randn(*self.w_shape).to(input.device) * 0.001)
                    )[1]
                    * pixels
                )
        if not reverse:
            weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
        else:
            try:
                weight = torch.inverse(self.weight.double()).float().view(w_shape[0], w_shape[1], 1, 1)
            except:
                weight = torch.inverse(
                    self.weight.double()
                    + (self.weight.mean() * torch.randn(*self.w_shape).to(input.device) * 0.001)
                    .float()
                    .view(w_shape[0], w_shape[1], 1, 1)
                )
        return weight, dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
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

    def forward(self, input: torch.Tensor, logdet=None, ft=None, reverse=False):
        # reverse -- True
        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            z = z + shiftFt
            z = z * scaleFt
            logdet = logdet + self.get_logdet(scaleFt)

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            # self.asserts(scale, shift, z1, z2)
            z2 = z2 + shift
            z2 = z2 * scale

            logdet = logdet + self.get_logdet(scale)
            z = torch.cat((z1, z2), dim=1)
            output = z
        else:
            # ===> Reach here
            z = input

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            # self.asserts(scale, shift, z1, z2)
            z2 = z2 / scale
            z2 = z2 - shift
            z = torch.cat((z1, z2), dim=1)
            logdet = logdet - self.get_logdet(scale)

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            z = z / scaleFt
            z = z - shiftFt
            logdet = logdet - self.get_logdet(scaleFt)

            output = z
        return output, logdet

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z, f):
        h = f(z)
        shift, scale = thops.split_cross(h)
        scale = torch.sigmoid(scale + 2.0) + self.affine_eps
        return scale, shift

    def feature_extract_aff(self, z1, ft, f):
        z = torch.cat([z1, ft], dim=1)
        h = f(z)
        shift, scale = thops.split_cross(h)
        scale = torch.sigmoid(scale + 2.0) + self.affine_eps
        return scale, shift

    def split(self, z):
        z1 = z[:, : self.channels_for_nn]
        z2 = z[:, self.channels_for_nn :]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels=64, kernel_hidden=1, n_hidden_layers=1):
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
        do_actnorm=True,
        weight_std=0.05,
    ):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=(not do_actnorm))
        # init weight with std
        self.weight.data.normal_(mean=0.0, std=weight_std)
        if not do_actnorm:
            self.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)
        self.do_actnorm = do_actnorm
        # do_actnorm = True

    def forward(self, input):
        x = super().forward(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class Conv2dZeros(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=[3, 3], stride=[1, 1], padding="same", logscale_factor=3):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        # logscale_factor
        self.logscale_factor = logscale_factor
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logs * self.logscale_factor)


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
