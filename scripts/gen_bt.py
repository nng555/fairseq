from fairseq.models.roberta import RobertaModel
from fairseq.models.lstm_classifier import LSTMClassifier
import argparse
import os
import sys
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig
from hydra import slurm_utils

@hydra.main(config_path='/h/nng/conf/robust/config.yaml')
def gen_neighborhood_labels(cfg: DictConfig):
    shard = cfg.gen.shard
    slurm_utils.symlink_hydra(cfg, os.getcwd())

    model_path = os.path.join('/h/nng/slurm', cfg.gen.model.date, slurm_utils.resolve_name(cfg.gen.model.name))
    if not os.path.exists(os.path.join(model_path, 'checkpoint_best.pt')):
        for f in sorted(os.listdir(model_path))[::-1]:
            if os.path.exists(os.path.join(model_path, f, 'checkpoint_best.pt')):
                model_path = os.path.join(model_path, f)
                break

    model_path = os.path.join(model_path, 'checkpoint_best.pt')

    base_path = '/h/nng/data'

    d_path = os.path.join(base_path, cfg.data.task, cfg.data.name)
    if cfg.data.fdset == 'iwslt':
        bpe_code_path = os.path.join(d_path, cfg.data.fdset, 'orig', 'code')
    else:
        bpe_code_path = os.path.join(base_path, cfg.data.task, 'domain-robustness', 'shared_models', cfg.data.src + cfg.data.tgt + '.' + cfg.data.fdset + '.bpe')
    bin_path = os.path.join(d_path, cfg.data.fdset, cfg.gen.model.bin, 'bin')
    f_path = os.path.join(d_path, cfg.data.fdset, cfg.data.bin, 'train.gen.' + cfg.data.src + '_' + str(shard))
    out_path = os.path.join(d_path, cfg.data.fdset, cfg.data.bin, 'train.gen.' + cfg.data.tgt + '_' + str(shard))
    bpe_path = '/h/nng/programs/subword-nmt/subword_nmt'

    cat_sh = ['cat', f_path]
    token_sh = ['sacremoses', 'tokenize', '-a', '-l', cfg.data.src, '-q']
    bpe_sh = ['python', bpe_path + '/apply_bpe.py', '-c', bpe_code_path]
    fair_sh = ['fairseq-interactive', bin_path, \
               '--path', model_path, \
               '-s', cfg.data.src, \
               '-t', cfg.data.tgt, \
               '--remove-bpe', \
               '--buffer-size', '1024', \
               '--max-tokens', '8000']
    if cfg.data.label == 'bt':
        fair_sh = fair_sh + ['--beam', '5']
    elif cfg.data.label == 'bt_sampling':
        fair_sh = fair_sh + ['--sampling', '--beam', '1', '--nbest', '1']
    grep_sh = ['grep', '^H-']
    cut_sh = ['cut', '-f', '3-']
    detoken_sh = ['sacremoses', 'detokenize', '-l', cfg.data.tgt, '-q']

    cat_p = subprocess.Popen(cat_sh, stdout=subprocess.PIPE)
    token_p = subprocess.Popen(token_sh, stdin=cat_p.stdout, stdout=subprocess.PIPE)
    cat_p.stdout.close()
    bpe_p = subprocess.Popen(bpe_sh, stdin=token_p.stdout, stdout=subprocess.PIPE)
    token_p.stdout.close()
    fair_p = subprocess.Popen(fair_sh, stdin=bpe_p.stdout, stdout=subprocess.PIPE)
    bpe_p.stdout.close()
    grep_p = subprocess.Popen(grep_sh, stdin=fair_p.stdout, stdout=subprocess.PIPE)
    fair_p.stdout.close()
    cut_p = subprocess.Popen(cut_sh, stdin=grep_p.stdout, stdout=subprocess.PIPE)
    grep_p.stdout.close()
    detoken_p = subprocess.Popen(detoken_sh, stdin=cut_p.stdout, stdout=subprocess.PIPE)
    cut_p.stdout.close()
    output, err = detoken_p.communicate()

    with open(out_path, 'wb') as of:
        of.write(output)

if __name__ == "__main__":
    gen_neighborhood_labels()
