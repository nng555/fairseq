# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from fairseq import metrics, utils
from fairseq.checkpoint_utils import load_model_ensemble
from fairseq.criterions import FairseqCriterion, register_criterion

from fairseq.models.roberta import RobertaModel

@register_criterion("sentence_prediction")
class SentencePredictionCriterion(FairseqCriterion):

    def __init__(self,
            task,
            classification_head_name,
            regression_target,
            soft_labels,
            num_classes,
            st_model_path,
            threshold):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.regression_target = regression_target
        self.soft_labels = soft_labels
        self.num_classes = num_classes
        self.threshold = threshold

        if st_model_path:
            model = load_model_ensemble([st_model_path], task=task)
            self.st_model = model[0][0].eval()
            self.st_model.cuda()
        else:
            self.st_model = None

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='sentence_classification_head',
                            help='name of the classification head to use')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, "classification_heads")
            and self.classification_head_name in model.classification_heads
        ), "model must provide sentence classification head for --criterion=sentence_prediction"

        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )
        preds = logits.max(dim=1)[1]

        #print("===================")
        if self.soft_labels:
            targets = model.get_targets(sample, [logits]).view(-1, self.num_classes).float()
            sample_size = targets.size()[0]
        else:
            targets = model.get_targets(sample, [logits]).view(-1)
            sample_size = targets.numel()
        #print('targets' + str(targets), flush=True)

        if self.st_model:
            with torch.no_grad():
                st_logits, _  = self.st_model(
                    **sample['net_input'],
                    features_only=True,
                    classification_head_name=self.classification_head_name,
                )
                probs = F.softmax(st_logits)
            #print('st_logits' + str(st_logits), flush=True)
            #print('probs' + str(probs), flush=True)
            if self.soft_labels:
                self_targets = probs.view(-1, self.num_classes).float()
                sample_size = self_targets.size()[0]
            else:
                _, self_targets = torch.max(probs, -1)
                sample_size = self_targets.numel()
            #print('self_targets' + str(self_targets), flush=True)

            if self.threshold:
                mask, _ = torch.max(probs, dim=-1)
                mask = mask > self.threshold
                self_targets = self_targets[mask]
                logits = logits[mask]

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if self.st_model:
            loss_targets = self_targets
        else:
            loss_targets = targets

        if not self.regression_target:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            #print('lprobs' + str(lprobs), flush=True)
            if self.soft_labels:
                loss = F.kl_div(
                    lprobs,
                    loss_targets,
                    reduction='none',
                )
                #loss = -targets * logits
                loss = loss.sum(-1)
                loss = mean_ds(loss)
            else:
                loss = F.nll_loss(
                    lprobs,
                    loss_targets,
                    reduction='sum',
                )
        else:
            logits = logits.view(-1).float()
            loss_targets = loss_targets.float()
            loss = F.mse_loss(logits, loss_targets, reduction='sum')

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }
        '''
        if not self.regression_target:
            preds = logits.argmax(dim=1)
            print("#####################")
            print(preds)
            print(targets)
            print("#####################")
            logging_output['ncorrect'] = (preds == targets).sum()
        '''

        if not self.regression_target:

            if self.soft_labels:
                targets = targets.max(dim=1)[1]

            logging_output.update(
                ncorrect=(preds == targets).sum().item()
            )


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
