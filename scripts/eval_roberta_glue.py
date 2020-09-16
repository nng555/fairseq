from fairseq.models.roberta import RobertaModel
from fairseq.models.lstm_classifier import LSTMClassifier
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
def evaluate(cfg: DictConfig):
    #slurm_utils.symlink_hydra(cfg, os.getcwd())

    if cfg.data.task in ['nli']:
        base_path = '/scratch/ssd001/datasets/'
    elif cfg.data.task in ['sentiment']:
        base_path = '/h/nng/data'
    else:
        raise Exception('task %s data path not found'.format(cfg.data.task))

    model_data_path = os.path.join(base_path, cfg.data.task, cfg.eval.model.data)
    eval_data_path = os.path.join(base_path, cfg.data.task, cfg.eval.data)
    model_path = os.path.join('/h/nng/slurm', cfg.eval.model.date, slurm_utils.resolve_name(cfg.eval.model.name))
    if not os.path.exists(os.path.join(model_path, 'checkpoint_best.pt')):
        for f in sorted(os.listdir(model_path))[::-1]:
            if os.path.exists(os.path.join(model_path, f, 'checkpoint_best.pt')):
                model_path = os.path.join(model_path, f)
                break

    dict_path = os.path.join(model_data_path, cfg.data.fdset, cfg.data.bin, 'bin')

    #if 'seed' in cfg.train:
    #    ckpt_file = ':'.join([str(seed) + '.pt' for seed in cfg.train.seed])
    #else:
    #    ckpt_file = 'checkpoint_best.pt'
    ckpt_file = 'checkpoint_best.pt'

    print(os.path.join(model_path, cfg.eval.model.date, slurm_utils.resolve_name(cfg.eval.model.name)))
    if cfg.data.task == 'nli':
        model = RobertaModel.from_pretrained(
            model_path,
            checkpoint_file=ckpt_file,
            data_name_or_path = dict_path
        )
    elif cfg.data.task == 'sentiment':
        model = LSTMClassifier.from_pretrained(
            model_path,
            checkpoint_file=ckpt_file,
            data_name_or_path = dict_path
        )

    label_fn = lambda label: model.task.label_dictionary.string(
        [label + model.task.label_dictionary.nspecial]
    )

    model.cuda()
    model.eval()

    nval, nsamples = 0, 0
    '''
    if cfg.eval.split == 'valid':
        if cfg.data.name == 'mnli':
            if cfg.data.tdset in ['fiction', 'government', 'slate', 'telephone', 'travel']:
                dev_name = 'dev_matched'
            elif cfg.data.tdset == 'processed':
                dev_name = 'dev'
            else:
                dev_name = 'dev_mismatched'
        elif cfg.data.name == 'anli':
            dev_name = 'dev'
        elif cfg.data.name == 'aws':
            dev_name = 'valid'
        else:
            dev_name = 'valid'
    elif cfg.eval.split == 'test':
        if cfg.data.name == 'mnli':
            if cfg.data.tdset in ['fiction', 'government', 'slate', 'telephone', 'travel']:
                dev_name = 'test_matched'
            elif cfg.data.tdset == 'processed':
                dev_name = 'test'
            else:
                dev_name = 'test_mismatched'
        else:
            dev_name = cfg.eval.split
    else:
        dev_name = cfg.eval.split
    '''

    with open(os.path.join(eval_data_path, cfg.data.tdset, 'orig', cfg.eval.split + '.raw.input0')) as input0f, \
            open(os.path.join(eval_data_path, cfg.data.tdset, 'orig', cfg.eval.split + '.raw.label')) as targetf:

        input0 = input0f.readlines()
        target = targetf.readlines()

        if cfg.data.task in ['nli']:
            input1f = open(os.path.join(eval_data_path, cfg.data.tdset, cfg.data.bin, cfg.eval.split + '.raw.input1'))
            input1 = input1f.readlines()
            files = [input0, input1, target]
        else:
            files = [input0, target]

        for ex in zip(*files):

            ex = [s.strip() for s in ex]

            # otherwise use the full sentence
            s1_tok = model.encode(ex[0].strip())

            if cfg.data.task in ['nli']:
                s2_tok = model.encode(ex[1].strip())
                tokens = torch.cat((s1_tok, s2_tok))
            else:
                tokens = s1_tok

            if cfg.data.name != 'mnli':
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
                prediction_label = label_fn(prediction)
                nval += int(prediction_label == ex[-1])

            nsamples += 1
        print(cfg.data.tdset + ' | Accuracy: ', float(nval)/float(nsamples))

if __name__ == '__main__':
    evaluate()
