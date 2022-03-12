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

        self.already_print_params_num = False

        self.heats = opt['val']['heats'] # -- None
        self.n_sample = opt['val']['n_sample'] # -- 4

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_Flow(opt, step).to(self.device)
        if opt['gpu_ids'] is not None and len(opt['gpu_ids']) > 0:
            if opt['dist']:
                self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
            elif len(opt['gpu_ids']) > 1:
                self.netG = DataParallel(self.netG, opt['gpu_ids'])
            else:
                self.netG.cuda()

    def to(self, device):
        self.device = device
        self.netG.to(device)


    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT

    def get_module(self, model):
        if isinstance(model, nn.DataParallel):
            return model.module
        else:
            return model

    def get_sr(self, lq, heat=None, seed=None, z=None, epses=None):
        return self.get_sr_with_z(lq, heat, seed, z, epses)[0]


    def get_sr_with_z(self, lq, heat=None, seed=None, z=None, epses=None):
        self.netG.eval()
        if heat is None:
            heat = 0
        z = self.get_z(heat, seed, batch_size=lq.shape[0], lr_shape=lq.shape) # if z is None and epses is None else z
        # heat -- 0, seed -- None, lq.size() -- [1, 6, 400, 600], z.size() -- [1, 192, 50, 75]
        with torch.no_grad():
            sr, logdet = self.netG(lr=lq, z=z, eps_std=heat, reverse=True, epses=epses)
        self.netG.train()
        return sr, z

    def get_z(self, heat, seed=None, batch_size=1, lr_shape=None):
        size = (batch_size, 3 * 8 * 8, 8, 8)
        return torch.zeros(size)


        # if seed: torch.manual_seed(seed)
        # # opt_get(self.opt, ['network_G', 'flow', 'split', 'enable']) -- False
        # if opt_get(self.opt, ['network_G', 'flow', 'split', 'enable']):
        #     C = self.get_module(self.netG).flowUpsamplerNet.C
        #     H = int(self.opt['scale'] * lr_shape[2] // self.get_module(self.netG).flowUpsamplerNet.scaleH)
        #     W = int(self.opt['scale'] * lr_shape[3] // self.get_module(self.netG).flowUpsamplerNet.scaleW)
        #     z = torch.normal(mean=0, std=heat, size=(batch_size, C, H, W)) if heat > 0 else torch.zeros(
        #         (batch_size, C, H, W))
        # else:
        #     # opt_get(self.opt, ['network_G', 'flow', 'L']) -- 3
        #     L = opt_get(self.opt, ['network_G', 'flow', 'L']) or 3
        #     fac = 2 ** L
        #     # self.get_module(self.netG).flowUpsamplerNet.scaleH -- 8.0
        #     # self.get_module(self.netG).flowUpsamplerNet.scaleW -- 8.0
        #     H = int(self.opt['scale'] * lr_shape[2] // self.get_module(self.netG).flowUpsamplerNet.scaleH)
        #     W = int(self.opt['scale'] * lr_shape[3] // self.get_module(self.netG).flowUpsamplerNet.scaleW)
        #     size = (batch_size, 3 * fac * fac, H, W)
        #     z = torch.normal(mean=0, std=heat, size=size) if heat > 0 else torch.zeros(size)
        # return z

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        pdb.set_trace()

        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        pdb.set_trace()

        if self.heats is not None:
            for heat in self.heats:
                for i in range(self.n_sample):
                    out_dict[('NORMAL', heat, i)] = self.fake_H[(heat, i)].detach()[0].float().cpu()
        else:
            out_dict['NORMAL'] = self.fake_H[(0, 0)].detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict


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
