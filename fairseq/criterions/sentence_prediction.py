# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('sentence_prediction')
class SentencePredictionCriterion(FairseqCriterion):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--save-predictions', metavar='FILE',
                            help='file to save predictions to')
        parser.add_argument('--classification-head-name',
                            default='sentence_classification_head',
                            help='name of the classification head to use')
        parser.add_argument('--masked-lm-weight',
                            default=0.5,
                            type=float,
                            help='amount to weight the LM loss with')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, 'classification_heads')
            and self.args.classification_head_name in model.classification_heads
        ), 'model must provide sentence classification head for --criterion=sentence_prediction'

        if self.args.masked_lm_target:
            masked_tokens = sample['masked_target'].ne(self.padding_idx)
            lm_sample_size = masked_tokens.int().sum().item()

            if lm_sample_size == 0:
                masked_tokens = None

            logits, lm_logits = model(
                **sample['net_input'],
                features_only=True,
                return_masked=True,
                classification_head_name=self.args.classification_head_name,
                masked_tokens=masked_tokens,
            )
        else:
            logits, _ = model(
                **sample['net_input'],
                features_only=True,
                classification_head_name=self.args.classification_head_name,
            )

        if self.args.soft_labels:
            targets = model.get_targets(sample, [logits]).view(-1, self.args.num_classes).float()
            sample_size = targets.size()[0]
        else:
            targets = model.get_targets(sample, [logits]).view(-1)
            sample_size = targets.numel()

        if self.args.masked_lm_target:
            masked_targets = sample['masked_target'][masked_tokens]

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if not self.args.regression_target:
            if self.args.ordinal:
                #print(F.sigmoid(logits))
                #print(targets)
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    targets,
                    reduction='sum'
                )
            else:
                logits = F.log_softmax(logits, dim=-1, dtype=torch.float32)
                if self.args.soft_labels:
                    loss = F.kl_div(
                        logits,
                        targets,
                        reduction='none',
                    )
                    #loss = -targets * logits
                    loss = loss.sum(-1)
                    loss = mean_ds(loss)
                else:
                    loss = F.nll_loss(
                        logits,
                        targets,
                        reduction='sum',
                    )
        else:
            logits = logits.squeeze().float()
            targets = targets.float()
            loss = F.mse_loss(
                logits,
                targets,
                reduction='sum',
            )

        if self.args.masked_lm_target:
            loss = loss * (1 - self.args.masked_lm_weight) + \
                self.args.masked_lm_weight * F.nll_loss(
                F.log_softmax(
                    lm_logits.view(-1, lm_logits.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                ),
                masked_targets.view(-1),
                reduction='sum',
                ignore_index=self.padding_idx,
            )

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }

        if not self.args.regression_target:
            if self.args.ordinal:
                logits = ord_to_prob(F.sigmoid(logits))
                targets = ord_to_prob(targets)
            preds = logits.max(dim=1)[1]

            if self.args.soft_labels:
                targets = targets.max(dim=1)[1]

            logging_output.update(
                ncorrect=(preds == targets).sum().item()
            )
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            agg_output.update(accuracy=ncorrect/nsentences)

        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output

def ord_to_prob(probits):
    logits = probits - torch.cat((probits[:,1:], torch.zeros_like(probits)[:,0].unsqueeze(1)), dim=-1)
    return logits.float()
