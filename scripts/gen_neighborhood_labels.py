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
import math

import hydra
from omegaconf import DictConfig
from hydra import slurm_utils

def load_recon_model(recon, recon_date=None, recon_name=None, recon_rdset=None, d_path=None):
    if recon == "base" or not recon:
        r_model = RobertaModel.from_pretrained(
            '/scratch/hdd001/home/nng/roberta/roberta.base/',
            checkpoint_file = 'model.pt',
            data_name_or_path = '/scratch/hdd001/home/nng/roberta/roberta.base/'
        )
        r_encode = r_model
    elif recon == 'large':
        r_model = RobertaModel.from_pretrained(
            '/scratch/hdd001/home/nng/roberta/roberta.large/',
            checkpoint_file = 'model.pt',
            data_name_or_path = '/scratch/hdd001/home/nng/roberta/roberta.large/'
        )
        r_encode = r_model
    elif recon == 'distil':
        r_encode = RobertaModel.from_pretrained(
            '/scratch/hdd001/home/nng/roberta/roberta.base/',
            checkpoint_file = 'model.pt',
            data_name_or_path = '/scratch/hdd001/home/nng/roberta/roberta.base/'
        )
        r_encode.eval()
        r_encode.cuda()
        r_model = AutoModelWithLMHead.from_pretrained('distilroberta-base')
        tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    elif recon == 'german':
        r_model = AutoModelWithLMHead.from_pretrained('bert-base-german-cased')
        tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
    elif recon == 'multilingual':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        r_model = AutoModelWithLMHead.from_pretrained("bert-base-multilingual-cased")
    elif recon == 'local':
        if not recon_date:
            r_file = os.path.join('/checkpoint/nng/keep', slurm_utils.resolve_name(recon_name))
        else:
            r_path = os.path.join('/h/nng/slurm', recon_date, slurm_utils.resolve_name(recon_name))
            found = False
            for f in sorted(os.listdir(r_path))[::-1]:
                if f == 'checkpoint_best.pt':
                    r_file = r_path
                    found = True
                    break
                if os.path.exists(os.path.join(r_path, f, 'checkpoint_best.pt')):
                    r_file = os.path.join(r_path, f)
                    found = True
                    break
            if not found:
                raise Exception("Model in path {} not found".format(r_path))

        print(r_file)
        print(os.path.join(d_path, recon_rdset, 'unlabelled', 'bin'))
        r_model = RobertaModel.from_pretrained(
            r_file,
            checkpoint_file = 'checkpoint_best.pt',
            data_name_or_path = os.path.join(d_path, recon_rdset, 'unlabelled', 'bin')
        )
        r_encode = r_model

    else:
        return None, None

    r_model.eval()
    if torch.cuda.is_available():
        r_model.cuda()
    return r_model, r_encode

@hydra.main(config_path=os.path.expandvars('$HOME/conf/$PROJ'), config_name='config')
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

    r_model, r_encode = load_recon_model(cfg.gen.recon, cfg.gen.recon_file.date, cfg.gen.recon_file.name, cfg.gen.recon_file.rdset, d_path)
    if r_model is None or r_encode is None:
        raise Exception("Model %s not found".format(cfg.gen.recon))

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

        if cfg.gen.noisy:

            # how many num_sample sets do we need to do
            num_probs = 1
            if cfg.gen.increment:
                num_probs += math.ceil((float(cfg.gen.max_mask_prob) - float(cfg.gen.mask_prob))/float(cfg.gen.increment))

            # process batch_num sentences at a time
            batch_num = int(len(sents)/cfg.gen.batch) + 1
            for b in range(batch_num):
                if b * cfg.gen.batch >= len(sents):
                    break

                # for every set of augmentations needed...
                sent_set = sents[b*cfg.gen.batch:(b+1)*cfg.gen.batch]
                pred_probs = [[] for _ in range(len(sent_set))]

                for i in range(cfg.gen.num_samples * num_probs):
                    toks = []
                    masks = []
                    if cfg.gen.increment:
                        mask_prob = cfg.gen.mask_prob + int(i/cfg.gen.num_samples) * cfg.gen.increment
                    else:
                        mask_prob = cfg.eval.mask_prob
                    for s in sent_set:
                        tok, mask = r_encode.masked_encode(
                                *s,
                                mask_prob=mask_prob,
                                random_token_prob=float(cfg.gen.random_token_prob),
                                leave_unmasked_prob=float(cfg.gen.leave_unmasked_prob),
                                sort=cfg.gen.sort
                        )
                        tok = tok[:model.max_positions[0]]
                        mask = mask[:model.max_positions[0]]
                        toks.append(tok)
                        masks.append(mask)

                    max_len = max([len(tok) for tok in toks])
                    pad_tok = r_encode.task.source_dictionary.pad()
                    toks = [F.pad(tok, (0, max_len - len(tok)), 'constant', pad_tok) for tok in toks]
                    masks = [F.pad(mask, (0, max_len - len(mask)), 'constant', pad_tok) for mask in masks]
                    toks = torch.stack(toks).cuda()
                    masks = torch.stack(masks).cuda()
                    if cfg.gen.recon in ['distil', 'german', 'multilingual']:
                        raise Exception("Not supported yet!")
                    else:
                        rec, rec_masks, rec_probn = r_model.reconstruction_prob_tok(toks, masks, reconstruct=True, topk=cfg.gen.topk)

                    for i in range(len(rec)):
                        pred_prob = model.predict(
                                        'sentence_classification_head',
                                        rec[i],
                                        return_logits=(cfg.train.regression_target or cfg.train.ordinal)
                                    ).cpu().detach().numpy()[0]
                        pred_probs[i].append(np.exp(pred_prob))

                print("=================")
                print(pred_probs, flush=True)
                pred_probs = [np.average(np.asarray(probs), axis=0) for probs in pred_probs]
                print(pred_probs, flush=True)
                print("=================")

                for prob in pred_probs:
                    label = label_fn(np.argmax(prob))
                    prob = [str(p) for p in prob]
                    softf.write(' '.join(prob) + '\n')
                    rawf.write(label + '\n')

        else:
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
