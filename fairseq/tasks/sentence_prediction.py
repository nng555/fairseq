# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import torch

import numpy as np
from fairseq import utils
from fairseq.data import (
    ConcatSentencesDataset,
    ConcatDataset,
    data_utils,
    Dictionary,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    OffsetTokensDataset,
    PateTeacherDataset,
    PrependTokenDataset,
    RawLabelDataset,
    ReconstructTokensDataset,
    RightPadDataset,
    RollDataset,
    SoftLabelDataset,
    SortDataset,
    StripTokenDataset,
    data_utils,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
import fairseq.checkpoint_utils as checkpoint_utils
from fairseq.models.roberta import RobertaModel
from fairseq.tasks import LegacyFairseqTask, register_task

from . import FairseqTask, register_task
from fairseq.data.encoders.utils import get_whole_word_mask

logger = logging.getLogger(__name__)


@register_task("sentence_prediction")
class SentencePredictionTask(LegacyFairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", metavar="FILE", help="file prefix for data")
        parser.add_argument(
            "--num-classes",
            type=int,
            default=-1,
            help="number of classes or regression targets",
        )
        parser.add_argument(
            "--init-token",
            type=int,
            default=None,
            help="add token at the beginning of each batch item",
        )
        parser.add_argument(
            "--separator-token",
            type=int,
            default=None,
            help="add separator token between inputs",
        )
        parser.add_argument("--regression-target", action="store_true", default=False)
        parser.add_argument("--no-shuffle", action="store_true", default=False)
        parser.add_argument(
            "--shorten-method",
            default="none",
            choices=["none", "truncate", "random_crop"],
            help="if not none, shorten sequences that exceed --tokens-per-sample",
        )
        parser.add_argument(
            "--shorten-data-split-list",
            default="",
            help="comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)',
        )
        parser.add_argument(
            "--add-prev-output-tokens",
            action="store_true",
            default=False,
            help="add prev_output_tokens to sample, used for encoder-decoder arch",
        )

        # augmentation arguments
        parser.add_argument('--augment', default='none',
                            choices=['none', 'mask', 'reconstruct'],
                            help='if not none, apply data augmentation')
        parser.add_argument('--depth', default=1, type=int,
                            help='depth of augmentation')
        parser.add_argument('--keep-original', action='store_true', default=False,
                            help='whether to keep the original examples when augmenting')
        parser.add_argument('--unlabelled-only', action='store_true', default=False,
                            help='whether to augment only unlabelled data')

        # masking arguments
        parser.add_argument('--mask-prob', default=0.0, type=float,
                            help='probability of replacing a token with mask')
        parser.add_argument('--leave-unmasked-prob', default=0.1, type=float,
                            help='probability that a masked token is unmasked')
        parser.add_argument('--random-token-prob', default=0.1, type=float,
                            help='probability of replacing a token with a random token')
        parser.add_argument('--freq-weighted-replacement', default=False, action='store_true',
                            help='sample random replacement words based on word frequencies')
        parser.add_argument('--mask-whole-words', default=False, action='store_true',
                            help='mask whole words; you may also want to set --bpe')
        parser.add_argument('--epoch-mask-rate', default=0.0, type=float,
                            help='rate at which to increase mask rate per epoch')
        parser.add_argument('--max-mask-rate', default=1.0, type=float,
                            help='maximum mask rate')

        # reconstruction arguments
        parser.add_argument('--recon-model-path', default='/scratch/hdd001/home/nng/roberta/roberta.base',
                            help='path to reconstruction model directory')
        #parser.add_argument('--recon-model-file', default='model.pt',
        #                    help='filename of reconstruction model checkpoint')
        #parser.add_argument('--recon-model-data', default='/scratch/hdd001/home/nng/roberta/roberta.base',
        #                    help='path to reconstruction model data directory')
        #parser.add_argument('--comp-model-path', default='/scratch/hdd001/home/nng/roberta/roberta.base',
        parser.add_argument('--comp-model-path', default=None,
                            help='path to comparison model directory')
        #parser.add_argument('--comp-model-file', default=None,
        #                    help='filename of comparison model checkpoint')
        #parser.add_argument('--comp-model-data', default='/scratch/hdd001/home/nng/roberta/roberta.base',
        #                    help='path to comparison model data directory')
        parser.add_argument('--topk', default=-1, type=int,
                            help='topk sampling for reconstruction')

        # self-training arguments
        #parser.add_argument('--self-train', default=False, action='store_true',
        #                    help='whether to self-train or not')
        parser.add_argument('--st-model-path', default='/scratch/hdd001/home/nng/roberta/roberta.base',
                            help='path to self-training model directory')
        #parser.add_argument('--st-model-file', default='model.pt',
        #                    help='filename of self-training model checkpoint')
        #parser.add_argument('--st-model-data', default='/scratch/hdd001/home/nng/roberta/roberta.base',
        #                    help='path to self-training model data directory')
        parser.add_argument('--threshold', default=None, type=float,
                            help='min probability for self-training prediction')

        # noisy student arguments
        parser.add_argument('--unlabelled-data', default=None,
                            help='path to the additional unlabelled data')
        parser.add_argument('--unlabelled-augment', default='none',
                            choices=['none', 'mask', 'reconstruct'],
                            help='if not none, apply data augmentation to unlabelled data')

        # random seed
        parser.add_argument('--data-seed', default=None, type=int,
                            help='random seed for data ordering')

        # PATE arguments
        parser.add_argument('--num-teachers', default=0, type=int,
                            help='total number of teachers to train in PATE')
        parser.add_argument('--teacher-idx', default=0, type=int,
                            help='index of teacher to train')

        parser.add_argument(
            '--max-source-positions', default=1024, type=int, metavar='N',
            help='max number of tokens in the source sequence'
        )
        parser.add_argument(
            '--max-target-positions', default=1024, type=int, metavar='N',
            help='max number of tokens in the target sequence'
        )
        parser.add_argument('--soft-labels', default=False, action='store_true',
                            help='use soft labels insetad of hard labels')
        parser.add_argument('--ordinal', default=False, action='store_true',
                            help='use soft labels with ordinal regression')

    def __init__(self, args, data_dictionary, label_dictionary):
        super().__init__(args)
        self.dictionary = data_dictionary

        # add mask token
        self.mask_idx = self.dictionary.add_symbol('<mask>')

        self._label_dictionary = label_dictionary
        if not hasattr(args, "max_positions"):
            self._max_positions = (
                args.max_source_positions,
                args.max_target_positions,
            )
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions

        self.depth = args.depth

        if hasattr(args, 'only_eval'):
            self.only_eval = args.only_eval
        else:
            self.only_eval = False

        if (self.args.augment == 'reconstruct' or self.args.unlabelled_augment == 'reconstruct') and not self.only_eval:
            print(self.args.recon_model_path)
            self.recon_model = checkpoint_utils.load_model_ensemble([self.args.recon_model_path])[0][0]
            self.recon_model.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))
            self.recon_model.eval()
            self.recon_model.cuda()

            self.comp_model=None
            print(self.args.comp_model_path)
            if self.args.comp_model_path:
                self.comp_model = checkpoint_utils.load_model_ensemble([self.args.comp_model_path])[0][0]
                self.comp_model.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))
                self.comp_model.eval()
                self.comp_model.cuda()

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol("<mask>")
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.num_classes > 0, "Must set --num-classes"

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, "input0", "dict.txt"),
            source=True,
        )
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        # load label dictionary
        if not args.regression_target:
            label_dict = cls.load_dictionary(
                args,
                os.path.join(args.data, "label", "dict.txt"),
                source=False,
            )
            logger.info("[label] dictionary: {} types".format(len(label_dict)))
        else:
            label_dict = data_dict
        return cls(args, data_dict, label_dict)

    def get_path(self, type, split, data_path):
        return os.path.join(data_path, type, split)

    def make_dataset(self, type, dictionary, split, data_path, combine):
        split_path = self.get_path(type, split, data_path)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        return dataset


    def build_dataset(self, input0, input1, split, augment, keep_original):

        src_mask = None

        # create masked input and targets
        mask_whole_words = get_whole_word_mask(self.args, self.source_dictionary) \
            if self.args.mask_whole_words else None

        if augment in ['mask', 'reconstruct'] and split == 'train' and self.args.mask_prob > 0 and not self.only_eval:
            input0_mask, input0_tgt = MaskTokensDataset.apply_mask(
                input0,
                self.source_dictionary,
                pad_idx=self.source_dictionary.pad(),
                mask_idx=self.mask_idx,
                seed=self.args.seed,
                mask_prob=self.args.mask_prob,
                leave_unmasked_prob=self.args.leave_unmasked_prob,
                random_token_prob=self.args.random_token_prob,
                epoch_mask_rate=self.args.epoch_mask_rate,
                max_mask_rate=self.args.max_mask_rate,
                freq_weighted_replacement=self.args.freq_weighted_replacement,
                mask_whole_words=mask_whole_words,
            )

            if input1 is not None:
                input1_mask, input1_tgt = MaskTokensDataset.apply_mask(
                    input1,
                    self.source_dictionary,
                    pad_idx=self.source_dictionary.pad(),
                    mask_idx=self.mask_idx,
                    seed=self.args.seed,
                    mask_prob=self.args.mask_prob,
                    leave_unmasked_prob=self.args.leave_unmasked_prob,
                    random_token_prob=self.args.random_token_prob,
                    epoch_mask_rate=self.args.epoch_mask_rate,
                    max_mask_rate=self.args.max_mask_rate,
                    freq_weighted_replacement=self.args.freq_weighted_replacement,
                    mask_whole_words=mask_whole_words,
                )

        if self.args.init_token is not None:
            input0 = PrependTokenDataset(input0, self.args.init_token)
            if augment in ['mask', 'reconstruct'] and split == 'train' and self.args.mask_prob > 0 and not self.only_eval:
                input0_mask = PrependTokenDataset(input0_mask, self.args.init_token)
                input0_tgt = PrependTokenDataset(input0_tgt, self.source_dictionary.pad())

        if input1 is None:
            src_tokens = input0
            if augment in ['mask', 'reconstruct'] and split == 'train' and self.args.mask_prob > 0 and not self.only_eval:
                src_mask = input0_mask
                src_tgt = input0_tgt
        else:
            if self.args.separator_token is not None:
                input1 = PrependTokenDataset(input1, self.args.separator_token)
                if augment in ['mask', 'reconstruct'] and split == 'train' and self.args.mask_prob > 0 and not self.only_eval:
                    input1_mask = PrependTokenDataset(input1_mask, self.args.separator_token)
                    input1_tgt = PrependTokenDataset(input1_tgt, self.source_dictionary.pad())

            src_tokens = ConcatSentencesDataset(input0, input1)
            if augment in ['mask', 'reconstruct'] and split == 'train' and self.args.mask_prob > 0 and not self.only_eval:
                src_mask = ConcatSentencesDataset(input0_mask, input1_mask)
                src_tgt = ConcatSentencesDataset(input0_tgt, input1_tgt)

        if augment == 'reconstruct' and split =='train' and self.args.mask_prob > 0 and not self.only_eval:
            src_mask = maybe_shorten_dataset(
                src_mask,
                split,
                self.args.shorten_data_split_list,
                self.args.shorten_method,
                self._max_positions,
                self.args.seed,
            )
            src_mask = ReconstructTokensDataset.apply_reconstruct(
                src_mask,
                src_tgt,
                pad_idx=self.source_dictionary.pad(),
                mask_idx=self.mask_idx,
                bos_idx=self.source_dictionary.bos(),
                eos_idx=self.source_dictionary.eos(),
                recon_model=self.recon_model,
                comp_model=self.comp_model,
                #device=self.recon_model.device,
                seed=self.args.seed,
                topk=self.args.topk,
            )

            # mask and reconstruct again for longer random walks
            if hasattr(self, 'depth'):
                for d in range(self.depth - 1):
                    src_mask, src_tgt = MaskTokensDataset.apply_mask(
                        src_mask,
                        self.source_dictionary,
                        pad_idx=self.source_dictionary.pad(),
                        mask_idx=self.mask_idx,
                        seed=self.args.seed,
                        mask_prob=self.args.mask_prob,
                        leave_unmasked_prob=self.args.leave_unmasked_prob,
                        random_token_prob=self.args.random_token_prob,
                        epoch_mask_rate=self.args.epoch_mask_rate,
                        max_mask_rate=self.args.max_mask_rate,
                        freq_weighted_replacement=self.args.freq_weighted_replacement,
                        mask_whole_words=mask_whole_words,
                        depth = d + 1,
                    )
                    src_mask = ReconstructTokensDataset.apply_reconstruct(
                        src_mask,
                        src_tgt,
                        pad_idx=self.source_dictionary.pad(),
                        mask_idx=self.mask_idx,
                        bos_idx=self.source_dictionary.bos(),
                        eos_idx=self.source_dictionary.eos(),
                        recon_model=self.recon_model.model,
                        comp_model=self.comp_model,
                        device=self.recon_model.device,
                        seed=self.args.seed,
                        topk=self.args.topk,
                        depth = d + 1,
                    )

            if keep_original:
                src_tokens = maybe_shorten_dataset(
                    src_tokens,
                    split,
                    self.args.shorten_data_split_list,
                    self.args.shorten_method,
                    self._max_positions,
                    self.args.seed,
                )
                src_mask = ConcatDataset([src_tokens, src_mask])

        if not src_mask:
            src_mask = src_tokens

        # return masked tokens if reconstructed, as well as the original tokens
        return src_mask, src_tokens

    def build_label_dataset(self, data_path, split, combine=False):
        label_path = "{0}.label".format(self.get_path('label', split, data_path))
        if not self.args.regression_target:
            if self.args.soft_labels:
                label_dataset = SoftLabelDataset([
                    [float(prob) for prob in x.strip().split()] for x in open(label_path).readlines()
                ], self.args.num_classes)
            else:
                label_dataset = self.make_dataset('label', self.target_dictionary, split, data_path, combine)
                if label_dataset is not None:
                    label_dataset = OffsetTokensDataset(
                        StripTokenDataset(
                            label_dataset,
                            id_to_strip=self.target_dictionary.eos(),
                        ),
                        offset=-self.target_dictionary.nspecial,
                    )
        else:
            if os.path.exists(label_path):

                def parse_regression_target(i, line):
                    values = line.split()
                    assert len(values) == self.args.num_classes, \
                        f'expected num_classes={self.args.num_classes} regression target values on line {i}, found: "{line}"'
                    return [float(x) for x in values]

                with open(label_path) as h:
                    label_dataset = RawLabelDataset([
                        parse_regression_target(i, line.strip())
                        for i, line in enumerate(h.readlines())
                    ])

        return label_dataset

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        input0 = self.make_dataset('input0', self.source_dictionary, split, self.args.data, combine)
        assert input0 is not None, 'could not find dataset: {}'.format(self.get_path(type, split, self.args.data))
        input1 = self.make_dataset('input1', self.source_dictionary, split, self.args.data, combine)
        src_tokens, src_orig = self.build_dataset(input0, input1, split, self.args.augment, self.args.keep_original)

        if self.args.unlabelled_data and split == 'train':
            unlabelled0 = self.make_dataset('input0', self.source_dictionary, split, self.args.unlabelled_data, combine)
            assert unlabelled0 is not None, 'could not find dataset: {}'.format(self.get_path(type, split, self.args.unlabelled_data))
            unlabelled1 = self.make_dataset('input1', self.source_dictionary, split, self.args.unlabelled_data, combine)
            unlabelled_tokens, unlabelled_orig = self.build_dataset(unlabelled0, unlabelled1, split, self.args.unlabelled_augment, False)
            src_tokens = ConcatDataset([src_tokens, unlabelled_tokens])

        if self.args.num_teachers and split == 'train':
            src_tokens = PateTeacherDataset(
                    src_tokens,
                    self.args.teacher_idx,
                    self.args.num_teachers,
            )

        if hasattr(self.args, 'data_seed'):
            with data_utils.numpy_seed(self.args.data_seed):
                shuffle = np.random.permutation(len(src_tokens))
        else:
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_tokens))

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                ),
                "src_lengths": NumelDataset(src_tokens, reduce=False),
            },
            "original_input": {
                "src_tokens": RightPadDataset(
                    src_orig,
                    pad_idx=self.source_dictionary.pad(),
                ),
                "src_lengths": NumelDataset(src_orig, reduce=False),
            },
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
        }

        if self.args.add_prev_output_tokens:
            prev_tokens_dataset = RightPadDataset(
                RollDataset(src_tokens, 1),
                pad_idx=self.dictionary.pad(),
            )
            dataset["net_input"].update(
                prev_output_tokens=prev_tokens_dataset,
            )

        label_dataset = self.build_label_dataset(self.args.data, split, combine)

        if self.args.keep_original and split == 'train':
            label_dataset = ConcatDataset([label_dataset, label_dataset])

        if self.args.unlabelled_data and split == 'train':
            unlabelled_dataset = self.build_label_dataset(self.args.unlabelled_data, split, combine)
            label_dataset = ConcatDataset([label_dataset, unlabelled_dataset])

        if self.args.num_teachers and split == 'train':
            label_dataset = PateTeacherDataset(
                    label_dataset,
                    self.args.teacher_idx,
                    self.args.num_teachers,
            )

        dataset.update(target=label_dataset)

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
        )

        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)

        model.register_classification_head(
            getattr(args, "classification_head_name", "sentence_classification_head"),
            num_classes=self.args.num_classes,
        )

        return model

    def max_positions(self):
        return self._max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary
