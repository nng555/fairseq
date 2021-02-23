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

@hydra.main(config_path='/h/nng/conf/robust/config.yaml')
def evaluate(cfg: DictConfig):
    #slurm_utils.symlink_hydra(cfg, os.getcwd())

    if cfg.data.task in ['nli']:
        base_path = '/scratch/ssd002/datasets/'
    elif cfg.data.task in ['sentiment']:
        base_path = '/h/nng/data'
    else:
        raise Exception('task %s data path not found'.format(cfg.data.task))

    d_path = os.path.join(base_path, cfg.data.task, cfg.data.name)
    print(d_path)
    print(cfg.data.fdset)
    print(cfg.gen.in_dset)

    # load the recon model
    if cfg.gen.seed is not None:
        torch.manual_seed(cfg.gen.seed)
        np.random.seed(cfg.gen.seed)


    r_model, r_encode = load_recon_model(cfg.gen.recon, cfg.gen.recon_file.date, cfg.gen.recon_file.name, cfg.gen.recon_file.rdset, d_path)
    if r_model is None or r_encode is None:
        raise Exception("Model %s not found".format(cfg.gen.recon))

    s_model, s_encode = load_recon_model(cfg.gen.comp, cfg.gen.comp_file.date, cfg.gen.comp_file.name, cfg.gen.comp_file.rdset, d_path)

    if cfg.gen.recon in ['german', 'multilingual']:
        softmax_mask = np.full(len(tokenizer.vocab), False)
        softmax_mask[tokenizer.all_special_ids] = True
        for k, v in tokenizer.vocab.items():
            if '[unused' in k:
                softmax_mask[v] = True

    # load the model
    model_data_path = os.path.join(base_path, cfg.data.task, cfg.eval.model.data)
    eval_data_path = os.path.join(base_path, cfg.data.task, cfg.eval.data)
    model_path = os.path.join('/h/nng/slurm', cfg.eval.model.date, slurm_utils.resolve_name(cfg.eval.model.name))
    if not os.path.exists(os.path.join(model_path, 'checkpoint_best.pt')):
        for f in sorted(os.listdir(model_path))[::-1]:
            if os.path.exists(os.path.join(model_path, f, 'checkpoint_best.pt')):
                model_path = os.path.join(model_path, f)
                break

    dict_path = os.path.join(model_data_path, cfg.data.fdset, cfg.data.bin, 'bin')
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

    if not cfg.eval.mask_prob:
        cfg.eval.mask_prob=0.15
    if not cfg.eval.random_token_prob:
        cfg.eval.random_token_prob=0.1
    if not cfg.eval.leave_unmasked_prob:
        cfg.eval.leave_unmasked_prob=0.1

    nval, nsamples = 0, 0

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

            if cfg.eval.noisy:
                i = 0
                rec_prob = None
                pred_probs = []
                skip = False
                while i < cfg.eval.num_samples:
                    toks = []
                    masks = []
                    if skip:
                        break
                    for _ in range(min(cfg.eval.num_samples - i, cfg.gen.batch)):
                        i += 1
                        if cfg.gen.recon in ['german', 'multilingual']:
                            tok, mask = hf_masked_encode(
                                    tokenizer,
                                    *ex[:-1],
                                    mask_prob=float(cfg.eval.mask_prob),
                                    random_token_prob=float(cfg.eval.random_token_prob),
                                    leave_unmasked_prob=float(cfg.eval.leave_unmasked_prob),
                            )
                        else:
                            tok, mask = r_encode.masked_encode(
                                    *ex[:-1],
                                    mask_prob=float(cfg.eval.mask_prob),
                                    random_token_prob=float(cfg.eval.random_token_prob),
                                    leave_unmasked_prob=float(cfg.eval.leave_unmasked_prob),
                                    sort=cfg.gen.sort
                            )
                        toks.append(tok)
                        masks.append(mask)

                    max_len = max([len(tok) for tok in toks])
                    if max_len > model.max_positions[0] or max_len > r_model.max_positions[0]:
                        skip = True
                        break
                    if cfg.gen.recon in ['german', 'multilingual']:
                        pad_tok = tokenizer.pad_token_id
                    else:
                        pad_tok = r_encode.task.source_dictionary.pad()
                    toks = [F.pad(tok, (0, max_len - len(tok)), 'constant', pad_tok) for tok in toks]
                    masks = [F.pad(mask, (0, max_len - len(mask)), 'constant', pad_tok) for mask in masks]
                    toks = torch.stack(toks).cuda()
                    masks = torch.stack(masks).cuda()
                    if cfg.gen.recon in ['distil', 'german', 'multilingual']:
                        raise Exception("Not supported yet!")
                    else:
                        rec, rec_masks, rec_probn = r_model.reconstruction_prob_tok(toks, masks, reconstruct=True, source_model=s_model, topk=cfg.gen.topk)

                    for j in range(len(rec)):
                        s_rec = r_model.decode(rec[j].cpu(), True)[0].strip()
                        #print(s_rec)

                    for tokens in rec:
                        pred_prob = model.predict(
                                        'sentence_classification_head',
                                        tokens,
                                        return_logits=(cfg.train.regression_target or cfg.train.ordinal)
                                    ).cpu().detach().numpy()[0]
                        pred_probs.append(pred_prob)

                    if rec_prob is not None:
                        rec_prob = torch.cat((rec_prob, rec_probn))
                    else:
                        rec_prob = rec_probn

                if skip:
                    continue

                pred_probs = torch.tensor(pred_probs)
                weights = F.softmax(rec_prob, dim=0)
                #print(F.softmax(pred_probs, dim=-1))
                #print(weights)
                weighted_preds = torch.sum(weights.unsqueeze(1) * pred_probs, axis=0)
                #print(F.softmax(weighted_preds, dim=0))

                if cfg.train.regression_target:
                    nval += (weighted_preds[0] - float(ex[-1]))**2
                else:
                    prediction = weighted_preds.argmax()
                    prediction_label = label_fn(prediction)
                    nval += int(prediction_label == ex[-1])
                    #print(str(nsamples) + ': ' + str(prediction_label == ex[-1]))

            else:
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
                    pred_prob = np.exp(pred_prob)/sum(np.exp(pred_prob))
                    #print(list(pred_prob))
                    prediction_label = label_fn(prediction)
                    nval += int(prediction_label == ex[-1])

            nsamples += 1
        print(cfg.data.tdset + ' | Accuracy: ', float(nval)/float(nsamples))

if __name__ == '__main__':
    evaluate()
