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

# xxxx1111  --> 1
class LLFlowModel(BaseModel):
    def __init__(self, opt):
        super(LLFlowModel, self).__init__(opt)

        self.opt = opt

        # define network and load pretrained models
        self.netG = networks.define_Flow(opt).to(self.device)

    def to(self, device):
        self.device = device
        self.netG.to(device)


    def get_sr(self, lq, heat=None):
        return self.get_sr_with_z(lq, heat)


    def get_sr_with_z(self, lq, heat=None):
        self.netG.eval()
        # heat -- 0, lq.size() -- [1, 6, 400, 600]
        with torch.no_grad():
            sr, logdet = self.netG(lr=lq, eps_std=heat)
        self.netG.train()
        return sr

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
