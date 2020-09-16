#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import logging
import os
import numpy as np
from typing import List, Dict, Iterator, Tuple, Any

import torch
from torch import nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.data import encoders


logger = logging.getLogger(__name__)


def from_pretrained(
    model_name_or_path,
    checkpoint_file='model.pt',
    data_name_or_path='.',
    archive_map=None,
    **kwargs
):
    from fairseq import checkpoint_utils, file_utils

    if archive_map is not None:
        if model_name_or_path in archive_map:
            model_name_or_path = archive_map[model_name_or_path]
        if data_name_or_path is not None and data_name_or_path in archive_map:
            data_name_or_path = archive_map[data_name_or_path]

        # allow archive_map to set default arg_overrides (e.g., tokenizer, bpe)
        # for each model
        if isinstance(model_name_or_path, dict):
            for k, v in model_name_or_path.items():
                if k == 'checkpoint_file':
                    checkpoint_file = v
                elif (
                    k != 'path'
                    # only set kwargs that don't already have overrides
                    and k not in kwargs
                ):
                    kwargs[k] = v
            model_name_or_path = model_name_or_path['path']

    model_path = file_utils.load_archive_file(model_name_or_path)

    # convenience hack for loading data and BPE codes from model archive
    if data_name_or_path.startswith('.'):
        kwargs['data'] = os.path.abspath(os.path.join(model_path, data_name_or_path))
    else:
        kwargs['data'] = file_utils.load_archive_file(data_name_or_path)
    for file, arg in {
        'code': 'bpe_codes',
        'bpecodes': 'bpe_codes',
        'sentencepiece.bpe.model': 'sentencepiece_model',
    }.items():
        path = os.path.join(model_path, file)
        if os.path.exists(path):
            kwargs[arg] = path

    if 'user_dir' in kwargs:
        utils.import_user_module(argparse.Namespace(user_dir=kwargs['user_dir']))

    models, args, task = checkpoint_utils.load_model_ensemble_and_task(
        [os.path.join(model_path, cpt) for cpt in checkpoint_file.split(os.pathsep)],
        arg_overrides=kwargs,
    )

    return {
        'args': args,
        'task': task,
        'models': models,
    }


class GeneratorHubInterface(nn.Module):
    """
    PyTorch Hub interface for generating sequences from a pre-trained
    translation or language model.
    """

    def __init__(self, args, task, models):
        super().__init__()
        self.args = args
        self.task = task
        self.models = nn.ModuleList(models)
        self.src_dict = task.source_dictionary
        self.tgt_dict = task.target_dictionary

        # optimize model for generation
        for model in self.models:
            model.prepare_for_inference_(args)

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(getattr(args, 'replace_unk', None))

        self.tokenizer = encoders.build_tokenizer(args)
        self.bpe = encoders.build_bpe(args)

        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(), *[model.max_positions() for model in models]
        )

        # this is useful for determining the device
        self.register_buffer('_float_tensor', torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    def translate(self, sentences: List[str], beam: int = 5, verbose: bool = False, **kwargs) -> List[str]:
        return self.sample(sentences, beam, verbose, **kwargs)

    def sample(self, sentences: List[str], beam: int = 1, verbose: bool = False, **kwargs) -> List[str]:
        if isinstance(sentences, str):
            return self.sample([sentences], beam=beam, verbose=verbose, **kwargs)[0]
        tokenized_sentences = [self.encode(sentence) for sentence in sentences]
        batched_hypos = self.generate(tokenized_sentences, beam, verbose, **kwargs)
        return [self.decode(hypos[0]['tokens']) for hypos in batched_hypos]

    def score(self, sentences: List[str], **kwargs):
        if isinstance(sentences, str):
            return self.score([sentences], **kwargs)[0]
        # NOTE: this doesn't support translation tasks currently
        tokenized_sentences = [self.encode(sentence) for sentence in sentences]
        return [hypos[0] for hypos in self.generate(tokenized_sentences, score_reference=True, **kwargs)]

    def generate(
        self,
        tokenized_sentences: List[torch.LongTensor],
        beam: int = 5,
        verbose: bool = False,
        skip_invalid_size_inputs=False,
        inference_step_args=None,
        **kwargs
    ) -> List[List[Dict[str, torch.Tensor]]]:
        if torch.is_tensor(tokenized_sentences) and tokenized_sentences.dim() == 1:
            return self.generate(
                tokenized_sentences.unsqueeze(0), beam=beam, verbose=verbose, **kwargs
            )[0]

        # build generator using current args as well as any kwargs
        gen_args = copy.copy(self.args)
        gen_args.beam = beam
        for k, v in kwargs.items():
            setattr(gen_args, k, v)
        generator = self.task.build_generator(self.models, gen_args)

        inference_step_args = inference_step_args or {}
        results = []
        for batch in self._build_batches(tokenized_sentences, skip_invalid_size_inputs):
            batch = utils.apply_to_sample(lambda t: t.to(self.device), batch)
            translations = self.task.inference_step(
                generator, self.models, batch, **inference_step_args
            )
            for id, hypos in zip(batch["id"].tolist(), translations):
                results.append((id, hypos))

        # sort output to match input order
        outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]

        if verbose:

            def getarg(name, default):
                return getattr(gen_args, name, getattr(self.args, name, default))

            for source_tokens, target_hypotheses in zip(tokenized_sentences, outputs):
                src_str_with_unk = self.string(source_tokens)
                logger.info('S\t{}'.format(src_str_with_unk))
                for hypo in target_hypotheses:
                    hypo_str = self.decode(hypo['tokens'])
                    logger.info('H\t{}\t{}'.format(hypo['score'], hypo_str))
                    logger.info('P\t{}'.format(
                        ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))
                    ))
                    if hypo['alignment'] is not None and getarg('print_alignment', False):
                        logger.info('A\t{}'.format(
                            ' '.join(['{}-{}'.format(src_idx, tgt_idx) for src_idx, tgt_idx in hypo['alignment']])
                        ))
        return outputs

    def predict(self, head: str, tokens: torch.LongTensor, return_logits: bool = False):
        features = self.extract_features(tokens.to(device=self.device))
        logits = self.models[0].classification_heads[head](features)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)

    def extract_features(self, tokens: torch.LongTensor, return_all_hiddens: bool = False) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > self.models[0].max_positions():
            raise ValueError('tokens exceeds maximum length: {} > {}'.format(
                tokens.size(-1), self.models[0].max_positions()
            ))
        features, extra = self.models[0](
            tokens.to(device=self.device),
            torch.tensor([len(t) for t in tokens], device=self.device),
            features_only=True,
            return_all_hiddens=return_all_hiddens,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra['inner_states']
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features  # just the last layer's features

    def encode(self, sentence: str, *addl_sentences, no_separator=False) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`) and we use an
        extra end-of-sentence (`</s>`) as a separator.

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> </s> 1 2 3 </s>`

        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::

            >>> roberta.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> roberta.encode(' world').tolist()
            [0, 232, 2]
            >>> roberta.encode('world').tolist()
            [0, 8331, 2]
        """
        bpe_sentence = '<s> ' + self.bpe.encode(sentence) + ' </s>'
        for s in addl_sentences:
            bpe_sentence += (' </s>' if not no_separator else '')
            bpe_sentence += ' ' + self.bpe.encode(s) + ' </s>'
        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False, add_if_not_exist=False)
        return tokens.long()

    def decode(self, tokens: torch.LongTensor, split_sent=False):
        assert tokens.dim() == 1
        tokens = tokens.numpy()
        tokens = np.delete(tokens, np.argwhere(tokens == self.task.source_dictionary.pad()))
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = (tokens == self.task.source_dictionary.eos())
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        if split_sent:
            sentences = np.split(tokens, eos_mask[:-1].nonzero()[0] + 1)
        else:
            sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [self.bpe.decode(self.task.source_dictionary.string(s)) for s in sentences]
        sentences = [s for s in sentences if s != '']
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def tokenize(self, sentence: str) -> str:
        if self.tokenizer is not None:
            sentence = self.tokenizer.encode(sentence)
        return sentence

    def detokenize(self, sentence: str) -> str:
        if self.tokenizer is not None:
            sentence = self.tokenizer.decode(sentence)
        return sentence

    def apply_bpe(self, sentence: str) -> str:
        if self.bpe is not None:
            sentence = self.bpe.encode(sentence)
        return sentence

    def remove_bpe(self, sentence: str) -> str:
        if self.bpe is not None:
            sentence = self.bpe.decode(sentence)
        return sentence

    def binarize(self, sentence: str) -> torch.LongTensor:
        return self.src_dict.encode_line(sentence, add_if_not_exist=False).long()

    def string(self, tokens: torch.LongTensor) -> str:
        return self.tgt_dict.string(tokens)

    def _build_batches(
        self, tokens: List[List[int]], skip_invalid_size_inputs: bool
    ) -> Iterator[Dict[str, Any]]:
        lengths = torch.LongTensor([t.numel() for t in tokens])
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.build_dataset_for_inference(tokens, lengths),
            max_tokens=self.args.max_tokens,
            max_sentences=self.args.max_sentences,
            max_positions=self.max_positions,
            ignore_invalid_inputs=skip_invalid_size_inputs,
            disable_iterator_cache=True,
        ).next_epoch_itr(shuffle=False)
        return batch_iterator


class BPEHubInterface(object):
    """PyTorch Hub interface for Byte-Pair Encoding (BPE)."""

    def __init__(self, bpe, **kwargs):
        super().__init__()
        args = argparse.Namespace(bpe=bpe, **kwargs)
        self.bpe = encoders.build_bpe(args)
        assert self.bpe is not None

    def encode(self, sentence: str) -> str:
        return self.bpe.encode(sentence)

    def decode(self, sentence: str) -> str:
        return self.bpe.decode(sentence)


class TokenizerHubInterface(object):
    """PyTorch Hub interface for tokenization."""

    def __init__(self, tokenizer, **kwargs):
        super().__init__()
        args = argparse.Namespace(tokenizer=tokenizer, **kwargs)
        self.tokenizer = encoders.build_tokenizer(args)
        assert self.tokenizer is not None

    def encode(self, sentence: str) -> str:
        return self.tokenizer.encode(sentence)

    def decode(self, sentence: str) -> str:
        return self.tokenizer.decode(sentence)
