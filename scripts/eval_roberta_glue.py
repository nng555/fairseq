from fairseq.models.roberta import RobertaModel
from fairseq.models.lstm_classifier import LSTMClassifier
from fairseq.models.fconv_classifier import FConvClassifier
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from hydra import slurm_utils

@hydra.main(config_path=os.path.expandvars('$HOME/conf/$PROJ'), config_name='config')
def evaluate_model(cfg: DictConfig):
    slurm_utils.symlink_hydra(cfg, os.getcwd())

    if cfg.data.task in ['nli', 'nng_dataset']:
        base_path = '/scratch/ssd002/datasets/'
    elif cfg.data.task in ['sentiment', 'translation', 'robust', 'pretrain']:
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

    dict_path = os.path.join(model_data_path, cfg.data.fdset, cfg.data.bin.name, 'bin')

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

    nval, nsamples = 0, 0

    with open(os.path.join(eval_data_path, cfg.data.tdset, 'orig', cfg.eval.split + '.raw.input0')) as input0f, \
            open(os.path.join(eval_data_path, cfg.data.tdset, 'orig', cfg.eval.split + '.raw.label')) as targetf:

        input0 = input0f.readlines()
        target = targetf.readlines()
        input1f = os.path.join(eval_data_path, cfg.data.tdset, cfg.data.bin.name, cfg.eval.split + '.raw.input1')

        if os.path.exists(input1f):
            input1 = open(input1f).readlines()
            files = [input0, input1, target]
        else:
            files = [input0, target]

        for ex in zip(*files):

            ex = [s.strip() for s in ex]

            # otherwise use the full sentence
            tokens = model.encode(ex[0].strip())

            for ex_tok in ex[1:-1]:
                ex_tok = model.encode(ex_tok)
                tokens = torch.cat((tokens, ex_tok))

            if len(tokens) > model.max_positions[0]:
                continue

            pred_prob = model.predict(
                            'sentence_classification_head',
                            tokens,
                            return_logits=(cfg.train.regression_target or cfg.train.ordinal)
                        ).cpu().detach().numpy()[0]

            if cfg.train.regression_target:
                nval += (pred_prob[0] - float(ex[-1]))**2
            else:
                if cfg.train.ordinal:
                    pred_prob = 1 / (1 + np.exp(-pred_prob))
                    pred_prob = pred_prob - np.pad(pred_prob[1:], (0,1), 'constant')
                prediction = pred_prob.argmax()
                pred_prob = np.exp(pred_prob)/sum(np.exp(pred_prob))
                print(list(pred_prob))
                prediction_label = label_fn(prediction)
                nval += int(prediction_label == ex[-1])
                print(str(nsamples) + ': ' + str(prediction_label == ex[-1]))

            nsamples += 1
        print(cfg.data.tdset + ' | Accuracy: ', float(nval)/float(nsamples))

if __name__ == '__main__':
    evaluate_model()
