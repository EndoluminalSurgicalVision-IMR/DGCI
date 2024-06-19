# -*- coding: utf-8 -*-

import os
import sys
import yaml
import configargparse
import re
import numpy as np

import torch
import modules, utils
from dcinet import DCI
import sdf_meshing


def parse_idx_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals]


p = configargparse.ArgumentParser()

p.add_argument('--logging_root', type=str, default='./recon', help='root for logging')
p.add_argument('--config', required=False, default='configs/generate_all/airway_dci_20240104.yml', help='configs.')
p.add_argument('--subject_idx', default='0-9', type=parse_idx_range,
               help='index of subject to generate')
p.add_argument('--level', type=float, default=0.0, help='level of iso-surface for marching cube')
p.add_argument('--resolution', type=int, default=512, help='resolution of iso-surface for marching cube')
opt = p.parse_args()
with open(os.path.join(opt.config), 'r') as stream:
    meta_params = yaml.safe_load(stream)
model = DCI(**meta_params)
state_dict = torch.load(meta_params['checkpoint_path'])
filtered_state_dict = {k: v for k, v in state_dict.items() if k.find('detach') == -1}
model.load_state_dict(filtered_state_dict)
model.cuda()
root_path = os.path.join(opt.logging_root, meta_params['experiment_name'])
utils.cond_mkdir(root_path)
for idx in opt.subject_idx:
    sdf_meshing.create_mesh(model,
                            os.path.join(root_path,'test%04d' % idx + 'stage2_reso' + str(opt.resolution) + '_level' + str(opt.level)),
                            subject_idx=idx,
                            N=opt.resolution,
                            level=opt.level)
