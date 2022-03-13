


import torch
from torch import nn as nn

import models.modules
import models.modules.Permutations
# from models.modules import flow, thops, FlowAffineCouplingsAblation
from models.modules import FlowAffineCouplingsAblation

# from utils.util import opt_get

import pdb

# xxxx1111
class FlowStep(nn.Module):
    # FlowPermutation = {
    #     "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
    #     "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
    #     "invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
    # }

    def __init__(self, in_channels, hidden_channels,
                 actnorm_scale=1.0, flow_permutation="invconv", flow_coupling="additive"):
        # check configures
        # assert flow_permutation in FlowStep.FlowPermutation, \
        #     "float_permutation should be in `{}`".format(FlowStep.FlowPermutation.keys())
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling

        self.norm_type = 'ActNorm2d'

        # 1. actnorm
        self.actnorm = models.modules.FlowActNorms.ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        self.invconv = models.modules.Permutations.InvertibleConv1x1(in_channels)

        # 3. coupling
        if flow_coupling == "CondAffineSeparatedAndCond":
            self.affine = models.modules.FlowAffineCouplingsAblation.CondAffineSeparatedAndCond(in_channels=in_channels)
        elif flow_coupling == "noCoupling":
            pass
        else:
            raise RuntimeError("coupling not Found:", flow_coupling)

    # def forward(self, input, logdet=None, reverse=False, rrdbResults=None):
    def forward(self, input, logdet=None, rrdbResults=None):
        # if reverse:
        #     return self.reverse_flow(input, logdet, rrdbResults)
        # else:
        #     return self.normal_flow(input, logdet, rrdbResults)
        return self.reverse_flow(input, logdet, rrdbResults)

    # def normal_flow(self, z, logdet, rrdbResults=None):
    #     # xxxx3333
    #     # if self.flow_coupling == "bentIdentityPreAct":
    #     #     z, logdet = self.bentIdentPar(z, logdet, reverse=False)

    #     # 1. actnorm
    #     # self.norm_type -- 'ActNorm2d'
    #     if self.norm_type == "ConditionalActNormImageInjector":
    #         z, logdet = self.actnorm(z, img_ft=rrdbResults, logdet=logdet, reverse=False)
    #     elif self.norm_type == "noNorm":
    #         pass
    #     else:
    #         z, logdet = self.actnorm(z, logdet=logdet, reverse=False)
    #     # z, logdet = self.actnorm(z, logdet=logdet, reverse=False)

    #     # 2. permute
    #     z, logdet = FlowStep.FlowPermutation[self.flow_permutation](self, z, logdet, False)
    #     need_features = self.affine_need_features() # False

    #     # 3. coupling
    #     if need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
    #         img_ft = getConditional(rrdbResults)
    #         z, logdet = self.affine(input=z, logdet=logdet, reverse=False, ft=img_ft)
    #     return z, logdet

    def reverse_flow(self, z, logdet, rrdbResults=None):

        need_features = self.affine_need_features() # True

        # 1.coupling
        # need_features:  True self.flow_coupling:  CondAffineSeparatedAndCond
        # need_features:  False self.flow_coupling:  noCoupling
        if need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
            z, logdet = self.affine(input=z, logdet=logdet, reverse=True, ft=rrdbResults)

        # 2. permute
        z, logdet = self.invconv(z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet

    def affine_need_features(self):
        need_features = False
        try:
            need_features = self.affine.need_features
        except:
            pass
        return need_features
