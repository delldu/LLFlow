


import numpy as np
import torch
from torch import nn as nn

import models.modules.Split
from models.modules import flow, thops
from models.modules.Split import Split2d
from models.modules.glow_arch import f_conv2d_bias
from models.modules.FlowStep import FlowStep
from utils.util import opt_get

import pdb

class FlowUpsamplerNet(nn.Module):
    def __init__(self, image_shape, hidden_channels, K, L=None,
                 actnorm_scale=1.0,
                 flow_permutation=None,
                 flow_coupling="affine"):

        super().__init__()

        # image_shape = (160, 160, 3)
        # hidden_channels = 64
        # K = 4
        # L = None
        # actnorm_scale = 1.0
        # flow_permutation = None
        # flow_coupling = 'CondAffineSeparatedAndCond'

        self.hr_size = 160 # opt['datasets']['train']['GT_size']
        self.layers = nn.ModuleList()
        self.output_shapes = []

        # self.sigmoid_output = opt['sigmoid_output'] if opt['sigmoid_output'] is not None else False
        # self.sigmoid_output -- False

        self.L = 3 # opt_get(opt, ['network_G', 'flow', 'L']) # 3
        self.K = 4 # opt_get(opt, ['network_G', 'flow', 'K']) # 4
        # if isinstance(self.K, int):
        #     self.K = [K for K in [K, ] * (self.L + 1)]
        self.K = [4, 4, 4, 4]

        # self.opt = opt
        H, W, self.C = image_shape
        # self.check_image_shape()

        self.levelToName = {
            # 0: 'fea_up4',
            1: 'fea_up2',
            2: 'fea_up1',
            3: 'fea_up0',
            # 4: 'fea_up-1'
        }

        affineInCh = 128 # self.get_affineInCh(opt_get) # 128
        flow_permutation = 'invconv' # self.get_flow_permutation(flow_permutation, opt) # 'invconv'

        # conditional_channels = {}
        # n_rrdb = 128 # self.get_n_rrdb_channels(opt, opt_get) # -- 128

        # conditional_channels[0] = n_rrdb
        # for level in range(1, self.L + 1):
        #     conditional_channels[level] = n_rrdb

        # Upsampler
        for level in range(1, self.L + 1):
            # 1. Squeeze
            H, W = self.arch_squeeze(H, W)

            # 2. K FlowStep
            self.arch_additionalFlowAffine(H, W, actnorm_scale, hidden_channels)
            self.arch_FlowStep(H, self.K[level], W, actnorm_scale, affineInCh, flow_coupling,
                               flow_permutation,
                               hidden_channels)
            # ,
            #                    n_conditinal_channels=conditional_channels[level])

        self.f = f_conv2d_bias(affineInCh, 2 * 3 * 64)

        self.H = H
        self.W = W        
        self.scaleH = 160 // H # -- 1
        self.scaleW = 160 // W # -- 1

    # def get_n_rrdb_channels(self, opt, opt_get):
    #     blocks = [1] # opt_get(opt, ['network_G', 'flow', 'stackRRDB', 'blocks'])
    #     n_rrdb = (len(blocks) + 1) * 64
    #     return n_rrdb

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
        # # 'additionalFlowNoAffine' in opt['network_G']['flow'] -- True
        # if 'additionalFlowNoAffine' in opt['network_G']['flow']:
        #     n_additionalFlowNoAffine = int(opt['network_G']['flow']['additionalFlowNoAffine'])
        #     # n_additionalFlowNoAffine -- 2, self.C -- 12
            
        #     for _ in range(n_additionalFlowNoAffine):
        #         self.layers.append(
        #             FlowStep(in_channels=self.C,
        #                      hidden_channels=64,
        #                      actnorm_scale=actnorm_scale,
        #                      flow_permutation='invconv',
        #                      flow_coupling='noCoupling', opt=opt))
        #         self.output_shapes.append([-1, self.C, H, W])

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

    # def get_flow_permutation(self, flow_permutation, opt):
    #     flow_permutation = opt['network_G']['flow'].get('flow_permutation', 'invconv')
    #     return flow_permutation # 'invconv'

    # def get_affineInCh(self, opt_get):
    #     affineInCh = [1] # opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'blocks']) or []
    #     # affineInCh ==> [1]
    #     affineInCh = (len(affineInCh) + 1) * 64
    #     return affineInCh # 128

    def check_image_shape(self):
        assert self.C == 1 or self.C == 3, ("image_shape should be HWC, like (64, 64, 3)"
                                            "self.C == 1 or self.C == 3")

    def forward(self, gt=None, rrdbResults=None, z=None, logdet=0., reverse=False, eps_std=None):
        # logdet = tensor([0.])
        # reverse = True
        # eps_std = 0

        # if reverse:
        #     sr, logdet = self.decode(rrdbResults, z, eps_std, logdet=logdet)
        #     return sr, logdet
        # else:
        #     assert gt is not None
        #     z, logdet = self.encode(gt, rrdbResults, logdet=logdet)

        #     return z, logdet

        sr, logdet = self.decode(rrdbResults, z, eps_std, logdet=logdet)
        return sr, logdet


    # def encode(self, gt, rrdbResults, logdet=0.0):
    #     fl_fea = gt
    #     reverse = False
    #     level_conditionals = {}
    #     bypasses = {}

    #     L = opt_get(self.opt, ['network_G', 'flow', 'L'])
    #     pdb.set_trace()

    #     for level in range(1, L + 1):
    #         bypasses[level] = torch.nn.functional.interpolate(gt, scale_factor=2 ** -level, mode='bilinear',
    #                                                           align_corners=False)

    #     for layer, shape in zip(self.layers, self.output_shapes):
    #         size = shape[2]
    #         level = int(np.log(self.hr_size / size) / np.log(2))
    #         if level > 0 and level not in level_conditionals.keys():
    #             if rrdbResults is None:
    #                 level_conditionals[level] = None
    #             else:
    #                 level_conditionals[level] = rrdbResults[self.levelToName[level]]

    #         if isinstance(layer, FlowStep):
    #             fl_fea, logdet = layer(fl_fea, logdet, reverse=reverse, rrdbResults=level_conditionals[level])
    #         else:
    #             fl_fea, logdet = layer(fl_fea, logdet, reverse=reverse)

    #     z = fl_fea

    #     return z, logdet

    #     # if not isinstance(epses, list):
    #     #     return z, logdet

    #     # epses.append(z)
    #     # return epses, logdet


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
                fl_fea, logdet = layer(fl_fea, logdet=logdet, reverse=True, rrdbResults=level_conditionals[level])
            else:
                fl_fea, logdet = layer(fl_fea, logdet=logdet, reverse=True)

        sr = fl_fea
        assert sr.shape[1] == 3
        return sr, logdet

