import os
import argparse
import numpy as np
import csv
from scipy import stats

import hydra
from omegaconf import DictConfig
from hydra import slurm_utils

@hydra.main(config_path='/h/nng/conf/robust/config.yaml', strict=False)
def display_bias(cfg: DictConfig):

    if cfg.extra:
        cfg.display.dir.name.append(cfg.extra)

    if cfg.gen.seed is not None:
        cfg.display.dir.name[3] = '_'.join(cfg.display.dir.name[3].split('_')[:-1])

    base_path = os.path.join('/h/nng/data', cfg.data.task, cfg.compare.name, cfg.data.tdset)
    csv_file = os.path.join(base_path, 'orig', cfg.eval.split + '.csv')

    # properties for each line
    bias_prop = []

    # possible values for each property
    prop_list = [[] for _ in range(len(cfg.display.props))]

    with open(csv_file, 'r') as csv_f:
        reader = csv.reader(csv_f)
        header = next(reader)
        for row in reader:
            res = []
            if cfg.display.prop_split == 'Race' and  row[header.index('Race')] == '':
                continue
            for i in range(len(cfg.display.props)):
                prop = cfg.display.props[i]
                if prop == "Person":
                    prop_val = row[header.index(prop)] if row[header.index("Race")] == "" else ""
                else:
                    prop_val = row[header.index(cfg.display.props[i])]
                res.append(prop_val)
                if prop_val not in prop_list[i]:
                    prop_list[i].append(prop_val)
            bias_prop.append(res)

    compare_idx = cfg.display.props.index(cfg.display.prop_split)

    if len(prop_list[compare_idx]) != 2:
        raise Exception("Can only compare 2 values")

    cfg.display.dir.name[2] = cfg.data.fdset
    cfg.display.dir.name[5] = cfg.data.tdset

    cfg.display.dir.name[4] = cfg.gen.seed
    t_tests = []
    for seed in empty_to_list(cfg.display.seed):
        cfg.display.dir.name[6] = seed

        display_dir = os.path.join('/h/nng/slurm', cfg.display.dir.date, slurm_utils.resolve_name(cfg.display.dir.name), 'log')
        fnames = sorted(os.listdir(display_dir))[::-1]
        for fname in fnames:
            if 'err' in fname:
                continue
            res = open(os.path.join(display_dir, fname), 'r').readlines()
            if res == [] or 'Accuracy' not in res[-1]:
                continue
            probs = []
            for line in res:
                line = line.strip()
                if line[0] == '[' and line[-1] == ']':
                    probs.append([float(val) for val in line[1:-1].split(',')])
            break

        x_dim_size = tuple(len(prop_l) for prop_l in prop_list)
        sums = np.zeros(x_dim_size)
        nums = np.zeros(x_dim_size)

        for i in range(len(bias_prop)):
            props = bias_prop[i]
            prop_idxs = [[prop_list[j].index(props[j])] for j in range(len(props))]
            sums[np.ix_(*prop_idxs)] += probs[i][0]
            nums[np.ix_(*prop_idxs)] += 1


        avgs = np.divide(sums, nums)

        avg_0 = avgs.take(0, axis=compare_idx).flatten()
        avg_1 = avgs.take(1, axis=compare_idx).flatten()

        avg_0 = avg_0[~np.isnan(avg_0)]
        avg_1 = avg_1[~np.isnan(avg_1)]

        t_tests.append(stats.ttest_ind(avg_0, avg_1))

    print(t_tests)


def empty_to_list(l):
    if l is None:
        return [None]
    else:
        return list(l)

if __name__ == "__main__":
    display_bias()
