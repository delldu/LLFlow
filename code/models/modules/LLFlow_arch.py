


import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.modules.RRDBNet_arch import RRDBNet
from models.modules.ConditionEncoder import ConEncoder1
from models.modules.FlowUpsamplerNet import FlowUpsamplerNet
import models.modules.thops as thops
import models.modules.flow as flow
from models.modules.color_encoder import ColorEncoder
from utils.util import opt_get
from models.modules.flow import unsqueeze2d, squeeze2d
from torch.cuda.amp import autocast

import pdb

class LLFlow(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4, K=None):
        super(LLFlow, self).__init__()
        # self.crop_size = 160 # opt['datasets']['train']['GT_size']
        # self.opt = opt

        self.RRDB = ConEncoder1(in_nc, out_nc, nf, nb, gc, scale)
        K = 4
        hidden_channels = 64

        self.flowUpsamplerNet = FlowUpsamplerNet((160, 160, 3), hidden_channels, K,
                             flow_coupling="CondAffineSeparatedAndCond")

        # if self.opt['align_maxpool']:
        self.max_pool = torch.nn.MaxPool2d(3)

    @autocast()
    def forward(self, gt=None, lr=None, z=None, eps_std=None, reverse=False, reverse_with_grad=False,
                lr_enc=None):

        with torch.no_grad():
            return self.reverse_flow(lr, z, eps_std=eps_std, lr_enc=lr_enc)
            
    def rrdbPreprocessing(self, lr):
        rrdbResults = self.RRDB(lr)

        block_idxs = [1] # opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'blocks']) or []
        low_level_features = [rrdbResults["block_{}".format(idx)] for idx in block_idxs]
        concat = torch.cat(low_level_features, dim=1)
        keys = ['last_lr_fea', 'fea_up1', 'fea_up2', 'fea_up4']
        if 'fea_up0' in rrdbResults.keys():
            keys.append('fea_up0')
        if 'fea_up-1' in rrdbResults.keys():
            keys.append('fea_up-1')
        for k in keys:
            h = rrdbResults[k].shape[2]
            w = rrdbResults[k].shape[3]
            rrdbResults[k] = torch.cat([rrdbResults[k], F.interpolate(concat, (h, w))], dim=1)

        return rrdbResults

    def get_score(self, disc_loss_sigma, z):
        score_real = 0.5 * (1 - 1 / (disc_loss_sigma ** 2)) * thops.sum(z ** 2, dim=[1, 2, 3]) - \
                     z.shape[1] * z.shape[2] * z.shape[3] * math.log(disc_loss_sigma)
        return -score_real

    def reverse_flow(self, lr, z, eps_std, lr_enc=None):

        logdet = torch.zeros_like(lr[:, 0, 0, 0])
        # pixels = thops.pixels(lr) * self.opt['scale'] ** 2  # self.opt['scale'] -- 1
        pixels = thops.pixels(lr)

        if lr_enc is None and self.RRDB:
            lr_enc = self.rrdbPreprocessing(lr)

        # if self.opt['cond_encoder'] == "NoEncoder": # False
        #     z = squeeze2d(lr[:,:3],8)
        # else:
        #     if 'avg_color_map' in self.opt.keys() and self.opt['avg_color_map']: # False
        #         z = squeeze2d(F.avg_pool2d(lr_enc['color_map'], 7, 1, 3), 8)
        #     else:
        #         z = squeeze2d(lr_enc['color_map'], 8)

        z = squeeze2d(lr_enc['color_map'], 8)
        x, logdet = self.flowUpsamplerNet(rrdbResults=lr_enc, z=z, eps_std=eps_std, reverse=True, logdet=logdet)
        return x, logdet
