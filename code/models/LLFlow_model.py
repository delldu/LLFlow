import logging
from collections import OrderedDict
from utils.util import get_resume_paths, opt_get

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from torch.cuda.amp import GradScaler, autocast

logger = logging.getLogger('base')

import pdb

class LLFlowModel(BaseModel):
    def __init__(self, opt, step):
        super(LLFlowModel, self).__init__(opt)

        self.opt = opt

        # define network and load pretrained models
        self.netG = networks.define_Flow(opt, step).to(self.device)

    def to(self, device):
        self.device = device
        self.netG.to(device)


    def get_sr(self, lq, heat=None, seed=None, z=None):
        return self.get_sr_with_z(lq, heat, seed, z)[0]


    def get_sr_with_z(self, lq, heat=None, seed=None, z=None):
        self.netG.eval()
        if heat is None:
            heat = 0
        z = self.get_z(heat, seed, batch_size=lq.shape[0], lr_shape=lq.shape) # if z is None and epses is None else z
        # heat -- 0, seed -- None, lq.size() -- [1, 6, 400, 600], z.size() -- [1, 192, 50, 75]
        with torch.no_grad():
            sr, logdet = self.netG(lr=lq, z=z, eps_std=heat, reverse=True)
        self.netG.train()
        return sr, z

    def get_z(self, heat, seed=None, batch_size=1, lr_shape=None):
        H = int(lr_shape[2] // self.netG.flowUpsamplerNet.scaleH)
        W = int(lr_shape[3] // self.netG.flowUpsamplerNet.scaleW)
        
        size = (batch_size, 3 * 8 * 8, H, W)
        z = torch.normal(mean=0, std=heat, size=size)
        return z

    def load(self):
        _, get_resume_model_path = get_resume_paths(self.opt)
        if get_resume_model_path is not None:
            self.load_network(get_resume_model_path, self.netG, strict=True, submodule=None)
            return

        load_path_G = self.opt['path']['pretrain_model_G']
        load_submodule = self.opt['path']['load_submodule'] if 'load_submodule' in self.opt['path'].keys() else 'RRDB'
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path'].get('strict_load', True),
                              submodule=load_submodule)

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
