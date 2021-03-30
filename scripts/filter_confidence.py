import os
import numpy as np
from numpy.random import laplace

import hydra
from omegaconf import DictConfig
from hydra import slurm_utils

@hydra.main(config_path=os.path.expandvars('$HOME/conf/$PROJ'), config_name='config')
def filter_confidence(cfg: DictConfig):
    base_path = '/h/nng/data'
    d_path = os.path.join(base_path, cfg.data.task, cfg.data.name, cfg.data.fdset, cfg.data.bin)

    with open(os.path.join(d_path, 'train.raw.input0'), 'r') as rawf, \
         open(os.path.join(d_path, 'train.gen.label'), 'r') as genf, \
         open(os.path.join(d_path, 'train.soft.label'), 'r') as softf, \
         open(os.path.join(d_path, 'train.raw.label'), 'r') as rawlf, \
         open(os.path.join(d_path, 'train.filter.input0'), 'w') as rawff, \
         open(os.path.join(d_path, 'train.gen.filter.label'), 'w') as genff, \
         open(os.path.join(d_path, 'train.soft.filter.label'), 'w') as softff, \
         open(os.path.join(d_path, 'train.raw.filter.label'), 'w') as rawlff:

        for rawl, genl, softl, rawll in zip(rawf, genf, softf, rawlf):
            scores = [float(val) for val in softl.strip().split()]
            if max(scores) > float(cfg.data.threshold):
                rawff.write(rawl)
                genff.write(genl)
                softff.write(softl)
                rawlff.write(rawll)

if __name__ == "__main__":
    filter_confidence()
