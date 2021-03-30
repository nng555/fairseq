import os
import argparse
import numpy as np
from tabulate import tabulate

import hydra
from omegaconf import DictConfig
from hydra import slurm_utils

@hydra.main(config_path=os.path.expandvars('$HOME/conf/$PROJ'), config_name='config')
def confusion(cfg: DictConfig):

    labels = ['entailment', 'neutral', 'contradiction']

    cfg.display.dir.name[2] = cfg.data.fdset
    cfg.display.dir.name[1] = cfg.slurm.eval.noise
    cfg.display.dir.name[4] = cfg.data.tdset
    cfg.display.dir.name[5] = cfg.display.seed

    display_dir = os.path.join('/h/nng/slurm', cfg.display.dir.date, slurm_utils.resolve_name(cfg.display.dir.name), 'log')
    if not os.path.exists(display_dir):
        print("Path {} does not exist!".format(display_dir))
    fnames = sorted(os.listdir(display_dir))[::-1]
    for fname in fnames:
        if 'err' in fname:
            continue
        res = open(os.path.join(display_dir, fname), 'r')
        confusion = [[0] * 3 for i in range(3)]
        for line in res:
            if line.strip() in labels:
                pred_idx = labels.index(line.strip())
                t_idx = labels.index(next(res).strip())
                confusion[pred_idx][t_idx] += 1

        print(tabulate([[labels[i]] + confusion[i] for i in range(3)], [""] + labels, tablefmt='github'))
        break



if __name__ == "__main__":
    confusion()
