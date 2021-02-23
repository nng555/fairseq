import os
import sys
import numpy as np
from operator import itemgetter

import hydra
from omegaconf import DictConfig
from hydra import slurm_utils

@hydra.main(config_path='/h/nng/conf/robust/config.yaml')
def filter_examples(cfg: DictConfig):
    shard = cfg.gen.shard
   # slurm_utils.symlink_hydra(cfg, os.getcwd())

    if cfg.data.task in ['nli']:
        base_path = '/scratch/ssd001/datasets/'
    elif cfg.data.task in ['sentiment']:
        base_path = '/h/nng/data'
    else:
        raise Exception('task %s data path not found'.format(cfg.data.task))

    d_path = os.path.join(base_path, cfg.data.task, cfg.data.name)

    with open(os.path.join(d_path, cfg.data.fdset, cfg.gen.dset, 'train.soft.label_' + str(shard)), 'r') as softf, \
         open(os.path.join(d_path, cfg.data.fdset, cfg.gen.dset, 'train.raw.label_' + str(shard)), 'r') as rawf, \
         open(os.path.join(d_path, cfg.data.fdset, cfg.gen.dset, 'train.gen.input0_' + str(shard)), 'r') as inputf:

        exs = []
        for row in softf:
            row = [float(val) for val in row.strip().split()]
            exs.append(row)

        for i, row in enumerate(rawf):
            exs[i].append(row.strip())

        for i, row in enumerate(inputf):
            exs[i].append(row.strip())

        exs_sort = sorted(exs, key=itemgetter(0))

    with open(os.path.join(d_path, cfg.data.fdset, cfg.gen.dset, 'train.soft.filter.label_' + str(shard)), 'w') as softff, \
         open(os.path.join(d_path, cfg.data.fdset, cfg.gen.dset, 'train.raw.filter.label_' + str(shard)), 'w') as rawff, \
         open(os.path.join(d_path, cfg.data.fdset, cfg.gen.dset, 'train.gen.filter.input0_' + str(shard)), 'w') as inputff:

        for row in exs[:cfg.gen.filter_num]:
            softff.write(' '.join(exs[:2]) + '\n')
            rawff.write(exs[3] + '\n')
            inputff.write(exs[4] + '\n')

        for row in exs[-cfg.gen.filter_num:]:
            softff.write(' '.join(exs[:2]) + '\n')
            rawff.write(exs[3] + '\n')
            inputff.write(exs[4] + '\n')

if __name__ == "__main__":
    filter_examples()
