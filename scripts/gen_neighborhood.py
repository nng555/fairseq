from fairseq.models.roberta import RobertaModel
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelWithLMHead, BertTokenizer

import hydra
from omegaconf import DictConfig
from hydra import slurm_utils

def hf_masked_encode(
        tokenizer,
        sentence: str,
        *addl_sentences,
        mask_prob=0.0,
        random_token_prob=0.0,
        leave_unmasked_prob=0.0):

    if random_token_prob > 0.0:
        weights = np.ones(len(tokenizer.vocab))
        weights[tokenizer.all_special_ids] = 0
        weights = weights / weights.sum()

    tokens = np.asarray(tokenizer.encode(sentence, *addl_sentences, add_special_tokens=True))

    if mask_prob == 0.0:
        return tokens

    sz = len(tokens)
    mask = np.full(sz, False)
    num_mask = int(mask_prob * sz + np.random.rand())

    mask_choice_p = np.ones(sz)
    for i in range(sz):
        if tokens[i] in [tokenizer.sep_token_id, tokenizer.cls_token_id, tokenizer.pad_token_id]:
            mask_choice_p[i] = 0
    mask_choice_p = mask_choice_p / mask_choice_p.sum()

    mask[np.random.choice(sz, num_mask, replace=False, p=mask_choice_p)] = True

    mask_targets = np.full(len(mask), tokenizer.pad_token_id)
    mask_targets[mask] = tokens[mask == 1]

    # decide unmasking and random replacement
    rand_or_unmask_prob = random_token_prob + leave_unmasked_prob
    if rand_or_unmask_prob > 0.0:
        rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
        if random_token_prob == 0.0:
            unmask = rand_or_unmask
            rand_mask = None
        elif leave_unmasked_prob == 0.0:
            unmask = None
            rand_mask = rand_or_unmask
        else:
            unmask_prob = leave_unmasked_prob / rand_or_unmask_prob
            decision = np.random.rand(sz) < unmask_prob
            unmask = rand_or_unmask & decision
            rand_mask = rand_or_unmask & (~decision)
    else:
        unmask = rand_mask = None

    if unmask is not None:
        mask = mask ^ unmask

    tokens[mask] = tokenizer.mask_token_id
    if rand_mask is not None:
        num_rand = rand_mask.sum()
        if num_rand > 0:
            tokens[rand_mask] = np.random.choice(
                len(tokenizer.vocab),
                num_rand,
                p=weights,
            )

    return torch.tensor(tokens).long(), torch.tensor(mask).long()

def hf_reconstruction_prob_tok(masked_tokens, target_tokens, tokenizer, model, softmax_mask, reconstruct=False, s_model=None, topk=1):
    single = False

    # expand batch size 1
    if masked_tokens.dim() == 1:
        single = True
        masked_tokens = masked_tokens.unsqueeze(0)
        target_tokens = target_tokens.unsqueeze(0)

    masked_fill = torch.ones_like(masked_tokens)

    masked_index = (target_tokens != tokenizer.pad_token_id).nonzero(as_tuple=True)
    masked_orig_index = target_tokens[masked_index]

    # edge case of no masked tokens
    if len(masked_orig_index) == 0:
        if reconstruct:
            return masked_tokens, masked_fill
        else:
            return 1.0

    masked_orig_enum = [list(range(len(masked_orig_index))), masked_orig_index]

    outputs = model(
        masked_tokens.long().to(device=next(model.parameters()).device),
    )

    features = outputs[0]
    logits = features[masked_index]
    probs = logits.softmax(dim=-1)

    if source_model:
        s_outputs = s_model(
            masked_tokens.long().to(device=next(model.parameters()).device),
        )
        s_features = outputs[0]
        s_logits = s_features[masked_index]
        s_probs = s_logits.softmax(dim=-1)
        probs = probs - s_probs
        probs -= torch.min(probs, dim=-1).values.unsqueeze(-1)

    for i in range(len(probs)):
        probs[i][softmax_mask] = 0

    if (reconstruct):

        recon_prob = 0

        # sample from topk if not unrestricted
        if topk != -1:
            values, indices = probs.topk(k=topk, dim=-1)
            probs = values.softmax(dim=-1)

        if (len(masked_index) > 1):
            tok_samples = [torch.multinomial(kprob, 1) for kprob in kprobs]
            samples = torch.cat([idx[tok] for idx, tok in zip(tok_samples, indices)])

            probs = [kprob[tok] for kprob, tok in zip(kprobs, tok_samples)]
            recon_prob += torch.sum(torch.log(probs)).item()
        else:
            tok_sample = torch.multinomial(kprobs, 1)
            samples = indices[tok_sample]
            recon_prob += torch.log(kprobs[tok_sample]).item()

        # set samples
        masked_tokens[masked_index] = samples
        masked_fill[masked_index] = samples

        if single:
            return masked_tokens[0], masked_fill[0], recon_prob
        else:
            return masked_tokens, masked_fill, recon_prob

    return torch.sum(torch.log(probs[masked_orig_enum])).item()


def load_recon_model(recon, recon_date=None, recon_name=None, recon_rdset=None, d_path=None):
    if recon == "base":
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

@hydra.main(config_path='/h/nng/conf/selftrain', config_name='config')
def gen_neighborhood(cfg: DictConfig):
    #slurm_utils.symlink_hydra(cfg, os.getcwd())
    shard = cfg.gen.shard
    if cfg.gen.seed is not None:
        torch.manual_seed(cfg.gen.seed)
        np.random.seed(cfg.gen.seed)

    if cfg.data.task in ['nli']:
        base_path = '/scratch/ssd001/datasets/'
    elif cfg.data.task in ['sentiment', 'translation']:
        base_path = '/h/nng/data'
    else:
        raise Exception('task %s data path not found'.format(cfg.data.task))

    d_path = os.path.join(base_path, cfg.data.task, cfg.data.name)
    print(d_path)
    print(cfg.data.fdset)
    print(cfg.gen.in_dset)
    f_path = os.path.join(d_path, cfg.data.fdset, cfg.gen.in_dset)

    r_model, r_encode = load_recon_model(cfg.gen.recon, cfg.gen.recon_file.date, cfg.gen.recon_file.name, cfg.gen.recon_file.rdset, d_path)
    if r_model is None or r_encode is None:
        raise Exception("Model %s not found".format(cfg.gen.recon))

    s_model, s_encode = load_recon_model(cfg.gen.comp, cfg.gen.comp_file.date, cfg.gen.comp_file.name, cfg.gen.comp_file.rdset, d_path)
    print(s_model)

    if cfg.gen.recon in ['german', 'multilingual']:
        softmax_mask = np.full(len(tokenizer.vocab), False)
        softmax_mask[tokenizer.all_special_ids] = True
        for k, v in tokenizer.vocab.items():
            if '[unused' in k:
                softmax_mask[v] = True

    ext0 = 'input0'
    ext1 = 'input1'
    extl = 'label'
    if cfg.data.task in ['translation']:
        ext0 = cfg.data.src
        extl = cfg.data.tgt

    #print(f_path)
    if cfg.gen.depth == 1:
        split = 'raw'
    else:
        split = 'gen'

    data_split = cfg.gen.split
    ext0_path = os.path.join(f_path, '.'.join([data_split, split, ext0]))
    ext1_path = os.path.join(f_path, '.'.join([data_split, split, ext1]))
    extl_path = os.path.join(f_path, '.'.join([data_split, 'raw', extl]))
    extid_path = os.path.join(f_path, '.'.join([data_split, 'id']))

    if os.path.exists(ext0_path):
        s0_file = [[s.strip()] for s in open(ext0_path, 'r').readlines()]
        shard_start = (int(len(s0_file)/cfg.gen.num_shards) + 1) * shard
        shard_end = (int(len(s0_file)/cfg.gen.num_shards) + 1) * (shard + 1)
        s0_file = s0_file[shard_start:shard_end]
        if len(s0_file) == 0:
            return
    else:
        s0_file = [[s.strip()] for s in open(ext0_path + '_' + str(shard), 'r').readlines()]

    if os.path.exists(extl_path):
        l_file = [s.strip() for s in open(extl_path, 'r').readlines()]
        shard_start = (int(len(l_file)/cfg.gen.num_shards) + 1) * shard
        shard_end = (int(len(l_file)/cfg.gen.num_shards) + 1) * (shard + 1)
        l_file = l_file[shard_start:shard_end]
    else:
        l_file = [s.strip() for s in open(extl_path + '_' + str(shard), 'r').readlines()]


    if os.path.exists(extid_path):
        id_file = [s.strip() for s in open(extid_path, 'r').readlines()]
        shard_start = (int(len(l_file)/cfg.gen.num_shards) + 1) * shard
        shard_end = (int(len(l_file)/cfg.gen.num_shards) + 1) * (shard + 1)
        id_file = id_file[shard_start:shard_end]
    else:
        extid_path = extid_path + '_' + str(shard)
        if os.path.exists(extid_path):
            id_file = [s.strip() for s in open(extid_path, 'r').readlines()]
        else:
            id_file = [i for i in range(len(open(ext0_path, 'r').readlines()))]


    if not os.path.exists(os.path.join(d_path, cfg.data.fdset, cfg.gen.dset)):
        os.makedirs(os.path.join(d_path, cfg.data.fdset, cfg.gen.dset))

    s0_rec_filename = os.path.join(d_path, cfg.data.fdset, cfg.gen.dset, data_split + '.gen.' + ext0)
    l_rec_filename = os.path.join(d_path, cfg.data.fdset, cfg.gen.dset, data_split + '.imp.' + extl)
    p_rec_filename = os.path.join(d_path, cfg.data.fdset, cfg.gen.dset, data_split + '.prob')
    id_rec_filename = os.path.join(d_path, cfg.data.fdset, cfg.gen.dset, data_split + '.id')
    if os.path.exists(s0_rec_filename) and not cfg.gen.overwrite:
        raise Exception("File already exists")
    if cfg.gen.num_shards != 1:
        s0_rec_filename += '_' + str(shard)
        l_rec_filename += '_' + str(shard)
        p_rec_filename += '_' + str(shard)
        id_rec_filename += '_' + str(shard)
    s0_rec_file = open(s0_rec_filename, 'w')
    l_rec_file = open(l_rec_filename, 'w')
    p_rec_file = open(p_rec_filename, 'w')
    id_rec_file = open(id_rec_filename, 'w')

    # load a second sentence input if task is nli
    if cfg.data.task in ['nli']:
        if os.path.exists(ext1_path):
            s1_file = [[s.strip()] for s in open(ext1_path, 'r').readlines()]
            shard_start = (int(len(s1_file)/cfg.gen.num_shards) + 1) * shard
            shard_end = (int(len(s1_file)/cfg.gen.num_shards) + 1) * (shard + 1)
            s1_file = s1_file[shard_start:shard_end]
        else:
            s1_file = [[s.strip()] for s in open(ext1_path + '_' + str(shard), 'r').readlines()]
        s1_rec_file = open(os.path.join(d_path, cfg.data.fdset, cfg.gen.dset, data_split + '.gen.' + ext1 + '_' + str(shard)), 'w')

    # sentences and labels to process
    sents = []
    probs = []
    l = []
    sent_ids = []

    # number sentences generated
    num_gen = []

    # sentence index to noise from
    gen_index = []

    # log probability of sentences for metropolis sampling
    probs = []

    # number of tries generating a new sentence
    num_tries = []

    # next sentence index to draw from
    next_sent = 0

    while(len(sents) < cfg.gen.batch):
        while True:
            if next_sent >= len(s0_file):
                break

            if cfg.data.task in ['nli']:
                next_len = len(r_encode.encode(s0_file[next_sent][0], s1_file[next_sent][0]))
            elif cfg.gen.recon in ['german', 'multilingual']:
                next_len = len(tokenizer.encode(s0_file[next_sent][0]))
            else:
                next_len = len(r_encode.encode(s0_file[next_sent][0]))

            if next_len > 4 and next_len < 505:
                break
            else:
                next_sent += 1

        if next_sent < len(s0_file):
            if cfg.data.task in ['nli']:
                sents.append(list(zip(s0_file[next_sent], s1_file[next_sent])))
            else:
                sents.append(list(zip(s0_file[next_sent])))
            l.append(l_file[next_sent])
            sent_ids.append(id_file[next_sent])
            probs.append([])

            # set initial metropolis probabilities
            num_gen.append(0)
            num_tries.append(0)
            gen_index.append(0)
            next_sent += 1

    while (sents != []):
        # remove any sentences that are done generating and dump to file
        for i in range(len(num_gen))[::-1]:
            if num_gen[i] == cfg.gen.num_samples or num_tries[i] > cfg.gen.max_tries:

                # dump sents to file and add new one
                gen_sents = sents.pop(i)
                gen_probs = probs.pop(i)
                num_gen.pop(i)
                if cfg.gen.metropolis:
                    probs.pop(i)
                gen_index.pop(i)
                label = l.pop(i)
                sent_id = sent_ids.pop(i)
                if cfg.gen.depth == 1:
                    skip = 1
                else:
                    skip = 0

                if cfg.gen.progressive and cfg.gen.only_last:
                    skip = -1

                # write generated sentences
                for sg, prob in zip(gen_sents[skip:], gen_probs):
                    s0_sent = repr(sg[0])[1:-1]
                    if cfg.gen.print_prob:
                        p_rec_file.write(str(prob) + '\n')
                    id_rec_file.write(str(sent_id) + '\n')
                    s0_rec_file.write(s0_sent + '\n')
                    if cfg.data.task in ['nli']:
                        s1_sent = repr(sg[1])[1:-1]
                        s1_rec_file.write(s1_sent + '\n')
                    l_rec_file.write(label + '\n')

                # get next sentence below 512
                while True:
                    if next_sent >= len(s0_file):
                        break

                    if cfg.data.task in ['nli']:
                        next_len = len(r_encode.encode(s0_file[next_sent][0], s1_file[next_sent][0]))
                    elif cfg.gen.recon in ['german', 'multilingual']:
                        next_len = len(tokenizer.encode(s0_file[next_sent][0]))
                    else:
                        next_len = len(r_encode.encode(s0_file[next_sent][0]))

                    if next_len > 4 and next_len < 505:
                        break
                    else:
                        next_sent += 1

                if next_sent < len(s0_file):
                    if cfg.data.task in ['nli']:
                        sents.append(list(zip(s0_file[next_sent], s1_file[next_sent])))
                    else:
                        sents.append(list(zip(s0_file[next_sent])))
                    # set initial metropolis probabilities
                    if cfg.gen.metropolis:
                        probs.append(get_log_p(' '.join(sents[-1][0])))
                    l.append(l_file[next_sent])
                    sent_ids.append(id_file[next_sent])
                    probs.append([])
                    num_gen.append(0)
                    num_tries.append(0)
                    if cfg.gen.metropolis:
                        gen_index.append(-1)
                    else:
                        gen_index.append(0)
                    next_sent += 1

        if len(sents) == 0:
            break

        toks = []
        masks = []
        #print('====================', flush=True)
        #print(num_gen, flush=True)
        #for v in sents:
        #    print(v, flush=True)

        # build batch
        for i in range(len(gen_index)):
            s = sents[i][gen_index[i]]
            if cfg.gen.recon in ['german', 'multilingual']:
                tok, mask = hf_masked_encode(
                        tokenizer,
                        *s,
                        mask_prob=cfg.gen.mask_prob,
                        random_token_prob=cfg.gen.random_token_prob,
                        leave_unmasked_prob=cfg.gen.leave_unmasked_prob,
                )
            else:
                tok, mask = r_encode.masked_encode(*s,
                        mask_prob=cfg.gen.mask_prob,
                        random_token_prob=cfg.gen.random_token_prob,
                        leave_unmasked_prob=cfg.gen.leave_unmasked_prob,
                        sort=cfg.gen.sort
                )
            toks.append(tok)
            masks.append(mask)

        # predict reconstruction
        max_len = max([len(tok) for tok in toks])
        if cfg.gen.recon in ['german', 'multilingual']:
            pad_tok = tokenizer.pad_token_id
        else:
            pad_tok = r_encode.task.source_dictionary.pad()

        toks = [F.pad(tok, (0, max_len - len(tok)), 'constant', pad_tok) for tok in toks]
        masks = [F.pad(mask, (0, max_len - len(mask)), 'constant', pad_tok) for mask in masks]
        toks = torch.stack(toks).cuda()
        masks = torch.stack(masks).cuda()
        if cfg.gen.recon in ['distil', 'german', 'multilingual']:
            rec, rec_masks, rec_prob = hf_reconstruction_prob_tok(toks, masks, tokenizer, r_model, softmax_mask, reconstruct=True, s_model=s_model, topk=cfg.gen.topk)
        else:
            rec, rec_masks, rec_prob = r_model.reconstruction_prob_tok(toks, masks, reconstruct=True, source_model=s_model, topk=cfg.gen.topk)

        # append to lists
        for i in range(len(rec)):
            if cfg.data.task in ['nli']:
                s_rec = [s.strip() for s in r_encode.decode(rec[i].cpu(), True)]
                s_rec = tuple(s_rec)
            else:
                if cfg.gen.recon in ['german', 'multilingual']:
                    rec_work = rec[i].cpu().tolist()
                    s_rec = tokenizer.decode(rec_work[:rec_work.index(tokenizer.sep_token_id)], skip_special_tokens=True)
                    s_rec = s_rec.replace("##", "")
                    s_rec = (s_rec,)
                else:
                    s_rec = r_encode.decode(rec[i].cpu(), False)
                    s_rec = (s_rec,)

            accept = False
            '''
            print('==========================')
            print(s_rec)
            print(sents[i])
            print('==========================', flush=True)
            '''
            if not cfg.gen.deduplicate or (s_rec not in sents[i] and '' not in s_rec):
                '''
                if cfg.gen.metropolis:
                    padding_mask = rec[i] != r_encode.task.source_dictionary.pad()
                    #pn = torch.sum(torch.log(r_model.sequence_probability(rec[i][padding_mask]))).item()
                    pn = get_log_p(' '.join(s_rec))
                    pforwards = r_model.reconstruction_prob_tok(r_encode.encode(*sents[i][gen_index[i]]), rec_masks[i][padding_mask])
                    pbackwards = r_model.reconstruction_prob_tok(rec[i][padding_mask], masks[i][padding_mask])
                    reject = np.exp((pforwards + pn) - (pbackwards + probs[i]))
                    if len(sents) < cfg.gen.batch:
                        print("====================")
                        print(s_rec)
                        print(sents[i][gen_index[i]])
                        print("{}\t{}\t{}\t{}".format(probs[i], pn, pforwards, pbackwards))
                        print(reject)
                        print("====================")
                    rand = np.random.random()
                    if rand < reject:
                        accept = True
                        probs[i] = pn
                else:
                '''
                accept = True


            if accept:
                sents[i].append(s_rec)
                probs[i].append(rec_prob)
                num_gen[i] += 1
                num_tries[i] = 0
                if cfg.gen.progressive or cfg.gen.metropolis:
                    gen_index[i] = -1
                else:
                    gen_index[i] = 0
            else:
                num_tries[i] += 1
                if cfg.gen.progressive:
                    gen_index[i] -= 1
                    if gen_index[i] < -len(sents[i]):
                        gen_index[i] = -1
                elif not cfg.gen.metropolis:
                    gen_index[i] += 1
                    if gen_index[i] == len(sents[i]):
                        gen_index[i] = 0
        del toks
        del masks

if __name__ == "__main__":
    gen_neighborhood()

