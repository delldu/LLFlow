
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.ConditionEncoder import ConEncoder1
from models.modules.FlowUpsamplerNet import FlowUpsamplerNet
# import models.modules.thops as thops
from models.modules.flow import squeeze2d

import pdb

# xxxx1111 --> 2, netG
class LLFlow(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=32, nb=4, gc=32, scale=1, K=4):
        super(LLFlow, self).__init__()
        self.RRDB = ConEncoder1(in_nc, out_nc, nf, nb, gc, scale)
        hidden_channels = 64
        self.flowUpsamplerNet = FlowUpsamplerNet((160, 160, 3), hidden_channels, K,
                             flow_coupling="CondAffineSeparatedAndCond")
        self.max_pool = nn.MaxPool2d(3)

    def forward(self, lr, eps_std):
        # lr.size()-- [1, 6, 400, 600]
        # z.size() -- [1, 192, 50, 75]

        # make noise tensor
        B, C, H, W = lr.shape
        size = (B, 3 * 8 * 8, H//8, W//8)
        z = torch.normal(mean=0, std=eps_std, size=size)

        logdet = torch.zeros_like(lr[:, 0, 0, 0])
        lr_enc = self.rrdbPreprocessing(lr)
        z = squeeze2d(lr_enc['color_map'], 8)
        x, logdet = self.flowUpsamplerNet(rrdbResults=lr_enc, z=z, eps_std=eps_std, logdet=logdet)
        return x, logdet

            
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

