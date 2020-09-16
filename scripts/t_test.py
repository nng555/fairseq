import os
import argparse
import time
import copy
import numpy as np
from scipy import stats

import hydra
import omegaconf
from omegaconf import DictConfig
from hydra import slurm_utils

@hydra.main(config_path='/h/nng/conf/robust/config.yaml', strict=False)
def display_results(cfg: DictConfig):
    res_string = ""
    std_string = ""
    if cfg.extra:
        cfg.display.dir.name.append(cfg.extra)

    cfg.display.dir.name[3] = '_'.join(cfg.display.dir.name[3].split('_')[:-1])

    for fdset in cfg.display.fdset:
        cfg.display.dir.name[2] = fdset
        for noise in empty_to_list(cfg.display.noise):
            cfg.display.dir.name[1] = noise
            row = []

            for tdset in cfg.display.tdset:
                cfg.display.dir.name[5] = tdset
                seed_res = []

                compare_dir = copy.deepcopy(cfg.display.dir.name)
                compare_dir[3] = cfg.display.compare.bin
                compare_res = []

                # check original eval
                for seed in empty_to_list(cfg.display.seed):
                    cfg.display.dir.name[6] = seed

                    for gen_seed in empty_to_list(cfg.gen.seed):
                        cfg.display.dir.name[4] = gen_seed
                        #print(slurm_utils.resolve_name(cfg.display.dir.name))
                        display_dir = os.path.join('/h/nng/slurm', cfg.display.dir.date, slurm_utils.resolve_name(cfg.display.dir.name), 'log')
                        #print(display_dir)
                        if not os.path.exists(display_dir):
                            #print("{} does not exist!".format(display_dir))
                            continue
                        fnames = sorted(os.listdir(display_dir))[::-1]
                        for fname in fnames:
                            if 'err' in fname:
                                continue
                            res = open(os.path.join(display_dir, fname), 'r').readlines()
                            if res != [] and 'Accuracy' in res[-1]:
                                seed_res.append(float(res[-1].rstrip().split(' ')[-1]))
                                break

                # check comparison eval
                for seed in empty_to_list(cfg.display.seed):

                    compare_dir[6] = seed
                    # check without any gen seed first
                    if cfg.display.compare.no_seed:
                        compare_dir[4] = None
                        display_dir = os.path.join('/h/nng/slurm', cfg.display.compare.date, slurm_utils.resolve_name(compare_dir), 'log')
                        if not os.path.exists(display_dir):
                            #print("{} does not exist!".format(display_dir))
                            continue
                        fnames = sorted(os.listdir(display_dir))[::-1]
                        for fname in fnames:
                            if 'err' in fname:
                                continue
                            res = open(os.path.join(display_dir, fname), 'r').readlines()
                            if res != [] and 'Accuracy' in res[-1]:
                                compare_res.append(float(res[-1].rstrip().split(' ')[-1]))
                                break
                    else:
                        for gen_seed in empty_to_list(cfg.gen.seed):
                            compare_dir[4] = gen_seed
                            display_dir = os.path.join('/h/nng/slurm', cfg.display.compare.date, slurm_utils.resolve_name(compare_dir), 'log')
                            if not os.path.exists(display_dir):
                                #print("{} does not exist!".format(display_dir))
                                continue
                            fnames = sorted(os.listdir(display_dir))[::-1]
                            for fname in fnames:
                                if 'err' in fname:
                                    continue
                                res = open(os.path.join(display_dir, fname), 'r').readlines()
                                if res != [] and 'Accuracy' in res[-1]:
                                    compare_res.append(float(res[-1].rstrip().split(' ')[-1]))
                                    break
                #print('====================')
                #print(seed_res)
                #print(compare_res)
                #print('====================')
                if seed_res == [] or compare_res == []:
                    row.append(1)
                    continue

                if len(seed_res) != 1:
                    t, p = stats.ttest_ind(seed_res, compare_res, equal_var = False)
                    row.append(p)
            res_string = res_string + '\t'.join([str(round(val, 4)) for val in row]) + '\n'
        res_string = res_string + '\n'
    print(res_string)

def empty_to_list(l):
    if l is None:
        return [None]
    else:
        return list(l)

if __name__ == "__main__":
    display_results()
