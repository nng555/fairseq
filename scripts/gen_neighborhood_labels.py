from fairseq.models.roberta import RobertaModel
from fairseq.models.lstm_classifier import LSTMClassifier
from fairseq.models.fconv_classifier import FConvClassifier
import argparse
import os
import sys
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

    model_path = os.path.join('/checkpoint/nng/keep', slurm_utils.resolve_name(cfg.gen.model.name))
    if not os.path.exists(model_path):
        model_path = os.path.join('/h/nng/slurm', cfg.gen.model.date, slurm_utils.resolve_name(cfg.gen.model.name))
        if not os.path.exists(os.path.join(model_path, 'checkpoint_best.pt')):
            for f in sorted(os.listdir(model_path))[::-1]:
                if os.path.exists(os.path.join(model_path, f, 'checkpoint_best.pt')):
                    model_path = os.path.join(model_path, f)
                    break

    if cfg.data.task in ['nli']:
        base_path = '/scratch/ssd002/datasets/'
    elif cfg.data.task in ['sentiment']:
        base_path = '/h/nng/data'
    else:
        raise Exception('task %s data path not found'.format(cfg.data.task))

    d_path = os.path.join(base_path, cfg.data.task, cfg.data.name)
    bin_path = os.path.join(d_path, cfg.gen.model.fdset, cfg.gen.model.bin, 'bin')
    print(bin_path)

    if cfg.data.task in ['nli']:
        model = RobertaModel.from_pretrained(
            model_path,
            checkpoint_file = 'checkpoint_best.pt',
            data_name_or_path = bin_path
        )
    elif cfg.data.task in ['sentiment']:
        model = RobertaModel.from_pretrained(
            model_path,
            checkpoint_file = 'checkpoint_best.pt',
            data_name_or_path = bin_path
        )
    model.eval()
    model.cuda()

    data_split = cfg.gen.split
    if data_split == 'unlabelled':
        i0path = os.path.join(d_path, cfg.data.fdset, cfg.gen.dset, 'train.raw.input0')
        i1path = os.path.join(d_path, cfg.data.fdset, cfg.gen.dset, 'train.raw.input1')
    else:
        i0path = os.path.join(d_path, cfg.data.fdset, cfg.gen.dset, data_split + '.gen.input0')
        i1path = os.path.join(d_path, cfg.data.fdset, cfg.gen.dset, data_split + '.gen.input1')
    print(i0path)

    if os.path.exists(i0path + '_' + str(shard)):
        s0_file = open(i0path + '_' + str(shard), 'r').readlines()
        if cfg.data.task in ['nli']:
            s1_file = open(i1path + '_' + str(shard), 'r').readlines()
            sents = [(s0.strip(), s1.strip()) for s0, s1 in zip(s0_file, s1_file)]
        else:
            sents = [(s0.strip(),) for s0 in s0_file]
    else:
        s0_file = open(i0path, 'r').readlines()
        shard_start = (int(len(s0_file)/cfg.gen.num_shards) + 1) * shard
        shard_end = (int(len(s0_file)/cfg.gen.num_shards) + 1) * (shard + 1)
        print(str(shard_start) + ':\t' + str(shard_end), flush=True)
        if cfg.data.task in ['nli']:
            s1_file = open(i1path, 'r').readlines()
            sents = [(s0.strip(), s1.strip()) for s0, s1 in zip(s0_file[shard_start:shard_end], s1_file[shard_start:shard_end])]
        else:
            sents = [(s0.strip(),) for s0 in s0_file[shard_start:shard_end]]

    label_fn = lambda label: model.task.label_dictionary.string(
        [label + model.task.label_dictionary.nspecial]
    )

    print(len(sents))
    if data_split == 'unlabelled':
        extra = []
        if cfg.data.teacher_idx is not None:
            extra.append(str(cfg.data.teacher_idx))
        if cfg.gen.num_shards != 1:
            extra.append(str(shard))
        softf_path = os.path.join(d_path, cfg.data.fdset, cfg.gen.dset, 'train.soft.label_' + '_'.join(extra))
        rawf_path = os.path.join(d_path, cfg.data.fdset, cfg.gen.dset, 'train.gen.label_' + '_'.join(extra))
    else:
        softf_path = os.path.join(d_path, cfg.data.fdset, cfg.gen.dset, data_split + '.soft.label_' + str(shard))
        rawf_path = os.path.join(d_path, cfg.data.fdset, cfg.gen.dset, data_split + '.raw.label_' + str(shard))

    if os.path.exists(rawf_path) and os.path.exists(softf_path) and not cfg.gen.overwrite:
        raise Exception("File already exists")

    with open(softf_path, 'w') as softf, \
         open(rawf_path, 'w') as rawf:
        batch_num = int(len(sents)/cfg.gen.batch) + 1
        for b in range(batch_num):
            if b * cfg.gen.batch >= len(sents):
                break
            toks = [model.encode(*s) for s in sents[b*cfg.gen.batch:(b+1)*cfg.gen.batch]]
            max_len = max([len(tok) for tok in toks])
            toks = [F.pad(tok, (0, max_len - len(tok)), 'constant', model.task.source_dictionary.pad()) for tok in toks]
            toks = torch.stack(toks)
            probs = model.predict(
                        'sentence_classification_head',
                        toks,
                        return_logits=(cfg.train.regression_target or cfg.train.ordinal)
                    ).cpu()
            del toks
            torch.cuda.empty_cache()

            if cfg.train.regression_target:
                for prob in probs:
                    rawf.write(str(prob.item()) + '\n')
            else:
                # write predicted to raw file
                _, idxs = torch.max(probs, -1)
                labels = [label_fn(idx) for idx in idxs]
                for label in labels:
                    rawf.write(label + '\n')

                # write softmax distribution to soft file
                if cfg.train.ordinal:
                    probs = F.sigmoid(probs).tolist()
                else:
                    probs = probs.softmax(dim=-1).tolist()
                for prob in probs:
                    prob = [str(p) for p in prob]
                    softf.write(' '.join(prob) + '\n')

if __name__ == "__main__":
    gen_neighborhood_labels()
