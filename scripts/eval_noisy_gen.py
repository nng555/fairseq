from fairseq.models.roberta import RobertaModel
from fairseq.models.lstm_classifier import LSTMClassifier
from fairseq.models.fconv_classifier import FConvClassifier
import argparse
import json
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import hydra
from operator import add
from omegaconf import DictConfig
from hydra import slurm_utils

@hydra.main(config_path=os.path.expandvars('$HOME/conf/$PROJ'), config_name='config')
def evaluate_model(cfg: DictConfig):
    slurm_utils.symlink_hydra(cfg, os.getcwd())

    if cfg.data.task in ['nli']:
        base_path = '/scratch/ssd001/datasets/'
    elif cfg.data.task in ['sentiment']:
        base_path = '/h/nng/data'
    else:
        raise Exception('task %s data path not found'.format(cfg.data.task))

    model_data_path = os.path.join(base_path, cfg.data.task, cfg.eval.model.data)
    eval_data_path = os.path.join(base_path, cfg.data.task, cfg.eval.data)
    if cfg.eval.model.date:
        model_path = os.path.join('/h/nng/slurm', cfg.eval.model.date, slurm_utils.resolve_name(cfg.eval.model.name))
        if not os.path.exists(os.path.join(model_path, 'checkpoint_best.pt')):
            for f in sorted(os.listdir(model_path))[::-1]:
                if os.path.exists(os.path.join(model_path, f, 'checkpoint_best.pt')):
                    model_path = os.path.join(model_path, f)
                    break
    else:
        model_path = os.path.join('/checkpoint/nng/keep', slurm_utils.resolve_name(cfg.eval.model.name))

    dict_path = os.path.join(model_data_path, cfg.data.fdset, cfg.eval.model.bin, 'bin')
    print(dict_path)

    ckpt_file = 'checkpoint_best.pt'

    print(model_path)
    if 'roberta' in cfg.train.arch:
        model = RobertaModel.from_pretrained(
            model_path,
            checkpoint_file=ckpt_file,
            data_name_or_path = dict_path
        )
    elif cfg.train.arch == 'fconv_classifier':
        model = FConvClassifier.from_pretrained(
            model_path,
            checkpoint_file=ckpt_file,
            data_name_or_path = dict_path
        )
    elif cfg.train.arch == 'lstm_classifier':
        model = LSTMClassifier.from_pretrained(
            model_path,
            checkpoint_file=ckpt_file,
            data_name_or_path = dict_path
        )
    else:
        raise Exception("Arch %s not supported".format(cfg.train.arch))

    label_fn = lambda label: model.task.label_dictionary.string(
        [label + model.task.label_dictionary.nspecial]
    )

    model.cuda()
    model.eval()



    # check for existing res json here

    with open(os.path.join(eval_data_path, cfg.data.tdset, cfg.data.bin.name, cfg.eval.split + '.gen.input0')) as input0f, \
            open(os.path.join(eval_data_path, cfg.data.tdset, cfg.data.bin.name, cfg.eval.split + '.id')) as idf, \
            open(os.path.join(eval_data_path, cfg.data.tdset, cfg.data.bin.name, cfg.eval.split + '.raw.label')) as targetf:

        input0 = input0f.readlines()
        target = targetf.readlines()
        ids = idf.readlines()

        if cfg.data.task in ['nli']:
            input1f = open(os.path.join(eval_data_path, cfg.data.tdset, cfg.data.bin.name, cfg.eval.split + '.raw.input1'))
            input1 = input1f.readlines()
            files = [input0, input1, target, ids]
        else:
            files = [input0, target, ids]

        res = {}
        targets = {}
        nsamples = 0

        j_dir = slurm_utils.get_j_dir(cfg)
        status_path = os.path.join(j_dir, 'eval_status.json')
        if os.path.exists(status_path):
            eval_status = json.load(open(status_path))
            res, targets, nsamples = eval_status
            files = [f[nsamples::] for f in files]

        batch_num = math.ceil(len(files[0])/cfg.eval.batch)
        for b in range(batch_num):

            bstart = b * cfg.eval.batch
            bend = (b+1) * cfg.eval.batch

            if bstart >= len(files[0]):
                break

            print("Processing {} to {}".format(str(bstart), str(bend)), flush=True)


            if cfg.data.task in ['nli']:
                toks = [model.encode(s1.strip(), s2.strip()) for s1, s2 in zip(input0[bstart:bend], input1[bstart:bend])]
            else:
                toks = [model.encode(s1.strip()) for s1 in input0[bstart:bend]]

            max_len = max([len(tok) for tok in toks])
            toks = [F.pad(tok, (0, max_len - len(tok)), 'constant', model.task.source_dictionary.pad()) for tok in toks]
            toks = torch.stack(toks).cuda()

            if cfg.data.name != 'mnli':
                if len(toks[0]) > model.max_positions[0]:
                    continue

            pred_probs = model.predict(
                            'sentence_classification_head',
                            toks,
                            return_logits=(cfg.train.regression_target or cfg.train.ordinal)
                        ).cpu().detach().numpy()

            for b_id, t in zip(ids[bstart:bend], target[bstart:bend]):
                b_id = b_id.strip()
                t = t.strip()
                if b_id not in res:
                    res[b_id] = [0.0 for _ in range(cfg.data.num_classes)]
                if b_id not in targets:
                    targets[b_id] = t


            for b_id, pred_prob in zip(ids[bstart:bend], pred_probs):
                b_id = b_id.strip()
                if cfg.train.regression_target:
                    res[b_id].append((pred_prob[0] - float(ex[-1]))**2)
                    res[b_id] += (pred_prob[0] - float(ex[-1]))**2
                else:
                    if cfg.train.ordinal:
                        pred_prob = 1 / (1 + np.exp(-pred_prob))
                        pred_prob = pred_prob - np.pad(pred_prob[1:], (0,1), 'constant')
                    pred_prob = np.exp(pred_prob)/sum(np.exp(pred_prob))
                    pred_prob= pred_prob.tolist()
                    res[b_id] = list(map(add, res[b_id], pred_prob))

            nsamples += len(pred_probs)

            with open(status_path, 'w') as statusf:
                json.dump([res, targets, nsamples], statusf)

        nval = 0

        for k in res:
            prediction = np.asarray(res[k]).argmax()
            prediction_label = label_fn(prediction)
            nval += int(prediction_label == targets[k])

        print(cfg.data.tdset + ' | Accuracy: ', float(nval)/len(res))

if __name__ == '__main__':
    evaluate_model()
