import os
import numpy as np
from numpy.random import laplace
from syft.frameworks.torch.dp import pate

import hydra
from omegaconf import DictConfig
from hydra import slurm_utils

@hydra.main(config_path='/h/nng/conf/robust/config.yaml')
def noisy_aggregate(cfg: DictConfig):

    base_path = '/h/nng/data'
    d_path = os.path.join(base_path, cfg.data.task, cfg.data.name, cfg.data.fdset, cfg.data.bin)
    print(d_path)

    label_dict = open(os.path.join(base_path, cfg.data.task, cfg.data.name, 'label.dict.txt'), 'r')
    labels = []
    for line in label_dict.readlines():
        label = line.strip().split()[0]
        if 'madeupword' not in label:
            labels.append(label)


    label_paths = []
    for f in os.listdir(d_path):
        if 'train.gen.label_' in f:
            label_paths.append(os.path.join(d_path, f))

    num_samples = len(open(label_paths[0], 'r').readlines())

    counts = [[0 for _ in range(len(labels))] for _ in range(num_samples)]
    teacher_preds = []

    for f in label_paths:
        with open(f, 'r') as lfile:
            teacher_preds.append([])
            for i, line in enumerate(lfile):
                idx = labels.index(line.strip())
                counts[i][idx] += 1
                teacher_preds[-1].append(idx)

    preds = []
    raw_preds = []
    for count in counts:
        count = [val + laplace(0.0, 1.0/float(cfg.gen.gamma)) for val in count]
        raw_preds.append(count.index(max(count)))
        preds.append(labels[count.index(max(count))])

    with open(os.path.join(d_path, 'train.gen.label'), 'w') as ofile:
        for pred in preds:
            ofile.write(pred + '\n')

    raw_preds = np.asarray(raw_preds)
    teacher_preds = np.asarray(teacher_preds)

    print(pate.perform_analysis(teacher_preds, raw_preds, 2 * float(cfg.gen.gamma), moments=15, delta=1e-05))


if __name__ == "__main__":
    noisy_aggregate()

