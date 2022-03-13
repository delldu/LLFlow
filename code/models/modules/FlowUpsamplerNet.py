import numpy as np
import torch
from torch import nn as nn

# import models.modules.Split
from models.modules import flow
#, thops
# from models.modules.Split import Split2d
from models.modules.glow_arch import f_conv2d_bias
from models.modules.FlowStep import FlowStep
# from utils.util import opt_get

import pdb

# xxxx1111
class FlowUpsamplerNet(nn.Module):
    def __init__(self, image_shape, hidden_channels, K, L=None,
                 actnorm_scale=1.0,
                 flow_permutation=None,
                 flow_coupling="affine"):

        super().__init__()

        self.hr_size = 160 # opt['datasets']['train']['GT_size']
        self.layers = nn.ModuleList()
        self.output_shapes = []

        self.L = 3 # opt_get(opt, ['network_G', 'flow', 'L']) # 3
        self.K = 4 # opt_get(opt, ['network_G', 'flow', 'K']) # 4
        # if isinstance(self.K, int):
        #     self.K = [K for K in [K, ] * (self.L + 1)]
        self.K = [4, 4, 4, 4]
        H, W, self.C = image_shape

        self.levelToName = {
            # 0: 'fea_up4',
            1: 'fea_up2',
            2: 'fea_up1',
            3: 'fea_up0',
            # 4: 'fea_up-1'
        }

        affineInCh = 128 # self.get_affineInCh(opt_get) # 128
        flow_permutation = 'invconv' # self.get_flow_permutation(flow_permutation, opt) # 'invconv'

        # Upsampler
        for level in range(1, self.L + 1):
            # 1. Squeeze
            H, W = self.arch_squeeze(H, W)

            # 2. K FlowStep
            self.arch_additionalFlowAffine(H, W, actnorm_scale, hidden_channels)
            self.arch_FlowStep(H, self.K[level], W, actnorm_scale, affineInCh, flow_coupling,
                               flow_permutation,
                               hidden_channels)

        self.f = f_conv2d_bias(affineInCh, 2 * 3 * 64)

        # self.H = H
        # self.W = W        
        # self.scaleH = 160 // H # -- 1
        # self.scaleW = 160 // W # -- 1

    def arch_FlowStep(self, H, K, W, actnorm_scale, affineInCh, flow_coupling, flow_permutation,
                      hidden_channels):

        for k in range(K):
            self.layers.append(
                FlowStep(in_channels=self.C,
                         hidden_channels=hidden_channels,
                         actnorm_scale=actnorm_scale,
                         flow_permutation=flow_permutation,
                         flow_coupling=flow_coupling))
            self.output_shapes.append([-1, self.C, H, W])


    def arch_additionalFlowAffine(self, H, W, actnorm_scale, hidden_channels):
        n_additionalFlowNoAffine = 2 # int(opt['network_G']['flow']['additionalFlowNoAffine'])
        for _ in range(n_additionalFlowNoAffine):
            self.layers.append(
                FlowStep(in_channels=self.C,
                         hidden_channels=hidden_channels,
                         actnorm_scale=actnorm_scale,
                         flow_permutation='invconv',
                         flow_coupling='noCoupling'))
            self.output_shapes.append([-1, self.C, H, W])


    def arch_squeeze(self, H, W):
        self.C, H, W = self.C * 4, H // 2, W // 2
        self.layers.append(flow.SqueezeLayer(factor=2))
        self.output_shapes.append([-1, self.C, H, W])
        return H, W

    def forward(self, rrdbResults=None, z=None, logdet=0., eps_std=None):
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

