from fairseq.models.roberta import RobertaModel
from fairseq.models.lstm_classifier import LSTMClassifier
from fairseq.criterions.sentence_prediction import ord_to_prob
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from hydra import slurm_utils

@hydra.main(config_path='/h/nng/conf/robust/config.yaml')
def class_noise(cfg: DictConfig):

    # get data path
    if cfg.data.task in ['nli']:
        base_path = '/scratch/ssd001/datasets/'
    elif cfg.data.task in ['sentiment']:
        base_path = '/h/nng/data'
    else:
        raise Exception('task %s data path not found'.format(cfg.data.task))

    # get model path
    model_path = os.path.join('/h/nng/slurm', cfg.gen.model.date, slurm_utils.resolve_name(cfg.gen.model.name))
    if not os.path.exists(os.path.join(model_path, 'checkpoint_best.pt')):
        for f in sorted(os.listdir(model_path))[::-1]:
            if os.path.exists(os.path.join(model_path, f, 'checkpoint_best.pt')):
                model_path = os.path.join(model_path, f)
                break
    d_path = os.path.join(base_path, cfg.data.task, cfg.data.name)
    bin_path = os.path.join(d_path, cfg.data.fdset, cfg.gen.model.bin, 'bin')
    f_path = os.path.join(d_path, cfg.data.fdset, cfg.gen.in_dset)
    print(f_path)

    # load model
    if cfg.data.task in ['nli']:
        model = RobertaModel.from_pretrained(
            model_path,
            checkpoint_file = 'checkpoint_best.pt',
            data_name_or_path = bin_path
        )
    elif cfg.data.task in ['sentiment']:
        model = LSTMClassifier.from_pretrained(
            model_path,
            checkpoint_file = 'checkpoint_best.pt',
            data_name_or_path = bin_path
        )

    label_fn = lambda label: model.task.label_dictionary.string(
        [label + model.task.label_dictionary.nspecial]
    )

    print(model.max_positions)
    model.cuda()
    model.eval()

    # load base roberta model
    r_base = RobertaModel.from_pretrained(
        '/scratch/hdd001/home/nng/roberta/roberta.base/',
        checkpoint_file = 'model.pt',
        data_name_or_path = '/scratch/hdd001/home/nng/roberta/roberta.base/'
    )
    print(r_base.max_positions)
    r_base.eval()
    r_base.cuda()

    # load input files
    if os.path.exists(os.path.join(f_path, '.'.join(['train.raw.input0']))):
        s0_file = [s.strip() for s in open(os.path.join(f_path, '.'.join(['train.raw.input0'])), 'r').readlines()]

    if os.path.exists(os.path.join(f_path, '.'.join(['train.raw.label']))):
        l_file = [s.strip() for s in open(os.path.join(f_path, '.'.join(['train.raw.label'])), 'r').readlines()]

    # load a second sentence input if task is nli
    if cfg.data.task in ['nli']:
        if os.path.exists(os.path.join(f_path, '.'.join(['train.raw.input1']))):
            s1_file = [s.strip() for s in open(os.path.join(f_path, '.'.join(['train.raw.input1'])), 'r').readlines()]
    else:
        s1_file = None

    sents = []
    l = []
    num_tries = []
    next_sent = 0

    next_sent = fill_sents(cfg, sents, l, num_tries, next_sent, r_base, model.max_positions[0], l_file, s0_file, s1_file)

    num_same = 0
    num_diff = 0

    while(sents != []):
        toks = []
        masks = []

        # build batch
        for i in range(len(sents)):
            s = sents[i]
            tok, mask = r_base.masked_encode(*s,
                    mask_prob=cfg.gen.mask_prob,
                    random_token_prob=cfg.gen.random_token_prob,
                    leave_unmasked_prob=cfg.gen.leave_unmasked_prob,
                    sort=cfg.gen.sort
            )
            toks.append(tok)
            masks.append(mask)

        # predict reconstruction
        max_len = max([len(tok) for tok in toks])
        toks = [F.pad(tok, (0, max_len - len(tok)), 'constant', r_base.task.source_dictionary.pad()) for tok in toks]
        masks = [F.pad(mask, (0, max_len - len(mask)), 'constant', r_base.task.source_dictionary.pad()) for mask in masks]
        toks = torch.stack(toks).cuda()
        masks = torch.stack(masks).cuda()
        rec, rec_masks = r_base.reconstruction_prob_tok(toks, masks, reconstruct=True, topk=cfg.gen.topk)
        preds = model.predict('sentence_classification_head', rec)

        remove = [False] * len(rec)
        for i in range(len(rec)):
            if cfg.data.task in ['nli']:
                s_rec = [s.strip() for s in r_base.decode(rec[i].cpu(), True)]
            else:
                s_rec = [r_base.decode(rec[i].cpu(), False)]
            # compare new label
            if s_rec != sents[i] and '' not in s_rec:
                remove[i] = True
                prediction = preds[i].argmax()
                pred_label = label_fn(prediction)
                if pred_label == l[i]:
                    num_same += 1
                else:
                    num_diff += 1
            else:
                num_tries[i] += 1
                if num_tries[i] > cfg.gen.max_tries:
                    remove[i] = True

        sents = [sents[i] for i in range(len(sents)) if not remove[i]]
        l = [l[i] for i in range(len(l)) if not remove[i]]
        next_sent = fill_sents(cfg, sents, l, num_tries, next_sent, r_base, model.max_positions[0], l_file, s0_file, s1_file)

    print(num_same)
    print(num_diff)

def fill_sents(cfg, sents, l, num_tries, next_sent, r_base, max_positions, l_file, s0_file, s1_file=None):
    while(len(sents) < cfg.gen.batch and next_sent < len(s0_file)):
        while True:
            if next_sent >= len(s0_file):
                break
            if s1_file is not None:
                next_len = len(r_base.encode(s0_file[next_sent], s1_file[next_sent]))
            else:
                next_len = len(r_base.encode(s0_file[next_sent]))

            if next_len < max_positions and next_len < r_base.max_positions[0]:
                break
            else:
                next_sent += 1

        if next_sent < len(s0_file):
            if cfg.data.task in ['nli']:
                sents.append([s0_file[next_sent], s1_file[next_sent]])
            else:
                sents.append([s0_file[next_sent]])
            l.append(l_file[next_sent])
            num_tries.append(0)
            next_sent += 1
    return next_sent



if __name__ == "__main__":
    class_noise()
