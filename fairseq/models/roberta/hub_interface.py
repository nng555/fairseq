# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data import encoders


class RobertaHubInterface(nn.Module):
    """A simple PyTorch Hub interface to RoBERTa.

    Usage: https://github.com/pytorch/fairseq/tree/master/examples/roberta
    """

    def __init__(self, cfg, task, model):
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.model = model
        self.max_positions = [self.model.max_positions()]

        self.bpe = encoders.build_bpe(cfg.bpe)

        # this is useful for determining the device
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    def masked_encode(self,
               sentence: str,
               *addl_sentences,
               mask_prob=0.0,
               random_token_prob=0.0,
               leave_unmasked_prob=0.0,
               sort=False,
               descending=False,
               no_separator=False) -> torch.LongTensor:
        start_tok = self.task.source_dictionary.encode_line('<s>', append_eos=False, add_if_not_exist=False)
        end_tok = self.task.source_dictionary.encode_line('</s>', append_eos=False, add_if_not_exist=False)
        pad_tok = torch.tensor([self.task.source_dictionary.pad_index])

        if random_token_prob > 0.0:
            weights = np.ones(len(self.task.source_dictionary))
            weights[:self.task.source_dictionary.nspecial] = 0
            weights = weights / weights.sum()

        def masked_encode_sent(sent):
            bpe_sentence = self.task.source_dictionary.encode_line(self.bpe.encode(sent), append_eos=False, add_if_not_exist=False)
            if (mask_prob == 0.0):
                return bpe_sentence

            # decide elements to mask
            sz = len(bpe_sentence)
            mask = np.full(sz, False)
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz + np.random.rand()
            )

            if sort:
                seq_probs = self.sequence_probability(torch.cat((start_tok, bpe_sentence, end_tok)).long().cuda())
                _, sort_idx = torch.sort(seq_probs[1:-1], descending=descending)
                sort_idx = sort_idx.cpu()
                mask[sort_idx[:num_mask]] = True
            else:
                mask[np.random.choice(sz, num_mask, replace=False)] = True

            mask_targets = np.full(len(mask), self.task.source_dictionary.pad_index)
            mask_targets[mask] = bpe_sentence[torch.from_numpy(mask.astype(np.uint8)) == 1]

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

            bpe_sentence[mask] = self.task.mask_idx
            if rand_mask is not None:
                num_rand = rand_mask.sum()
                if num_rand > 0:
                    bpe_sentence[rand_mask] = torch.from_numpy(np.random.choice(
                        len(self.task.source_dictionary),
                        num_rand,
                        p=weights,
                    )).int()
            return bpe_sentence, torch.from_numpy(mask_targets)

        bpe_sent, mask_targets = masked_encode_sent(sentence)
        bpe_sent = torch.cat((start_tok, bpe_sent, end_tok))
        mask_targets = torch.cat((pad_tok, mask_targets, pad_tok))

        for s in addl_sentences:
            if not no_separator:
                bpe_sent = torch.cat((bpe_sent, start_tok))
                mask_targets = torch.cat((mask_targets, pad_tok))
            tmp_sent, tmp_targets = masked_encode_sent(s)
            bpe_sent = torch.cat((bpe_sent, tmp_sent, end_tok))
            mask_targets = torch.cat((mask_targets, tmp_targets, pad_tok))
        return bpe_sent.long(), mask_targets.long()

    def encode(
        self, sentence: str, *addl_sentences, no_separator=False
    ) -> torch.LongTensor:
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
        bpe_sentence = "<s> " + self.bpe.encode(sentence) + " </s>"
        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator else ""
            bpe_sentence += " " + self.bpe.encode(s) + " </s>"
        tokens = self.task.source_dictionary.encode_line(
            bpe_sentence, append_eos=False, add_if_not_exist=False
        )
        return tokens.long()

    def decode(self, tokens: torch.LongTensor, split_sent=False):
        assert tokens.dim() == 1
        tokens = tokens.numpy()
        tokens = np.delete(tokens, np.argwhere(tokens == self.task.source_dictionary.pad()))
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        while(tokens[-1] == self.task.source_dictionary.eos()):
            tokens = tokens[:-1] # remove </s>
        eos_mask = (tokens == self.task.source_dictionary.eos())
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        if split_sent:
            sentences = np.split(tokens, eos_mask[:-1].nonzero()[0] + 1)
        else:
            sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [self.bpe.decode(self.task.source_dictionary.string(s)) for s in sentences]
        sentences = [s for s in sentences if s != '']
        if len(sentences) == 1 and not split_sent:
            return sentences[0]
        return sentences


    def extract_features(
        self, tokens: torch.LongTensor, return_all_hiddens: bool = False
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > self.model.max_positions():
            raise ValueError(
                "tokens exceeds maximum length: {} > {}".format(
                    tokens.size(-1), self.model.max_positions()
                )
            )
        features, extra = self.model(
            tokens.to(device=self.device),
            features_only=True,
            return_all_hiddens=return_all_hiddens,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra["inner_states"]
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features  # just the last layer's features

    def register_classification_head(
        self, name: str, num_classes: int = None, embedding_size: int = None, **kwargs
    ):
        self.model.register_classification_head(
            name, num_classes=num_classes, embedding_size=embedding_size, **kwargs
        )

    def predict(self, head: str, tokens: torch.LongTensor, return_logits: bool = False):
        features = self.extract_features(tokens.to(device=self.device))
        logits = self.model.classification_heads[head](features)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)

    def extract_features_aligned_to_words(
        self, sentence: str, return_all_hiddens: bool = False
    ) -> torch.Tensor:
        """Extract RoBERTa features, aligned to spaCy's word-level tokenizer."""
        from fairseq.models.roberta import alignment_utils
        from spacy.tokens import Doc

        nlp = alignment_utils.spacy_nlp()
        tokenizer = alignment_utils.spacy_tokenizer()

        # tokenize both with GPT-2 BPE and spaCy
        bpe_toks = self.encode(sentence)
        spacy_toks = tokenizer(sentence)
        spacy_toks_ws = [t.text_with_ws for t in tokenizer(sentence)]
        alignment = alignment_utils.align_bpe_to_words(self, bpe_toks, spacy_toks_ws)

        # extract features and align them
        features = self.extract_features(
            bpe_toks, return_all_hiddens=return_all_hiddens
        )
        features = features.squeeze(0)
        aligned_feats = alignment_utils.align_features_to_words(
            self, features, alignment
        )

        # wrap in spaCy Doc
        doc = Doc(
            nlp.vocab,
            words=["<s>"] + [x.text for x in spacy_toks] + ["</s>"],
            spaces=[True]
            + [x.endswith(" ") for x in spacy_toks_ws[:-1]]
            + [True, False],
        )
        assert len(doc) == aligned_feats.size(0)
        doc.user_token_hooks["vector"] = lambda token: aligned_feats[token.i]
        return doc

    def fill_mask(self, masked_input: str, topk: int = 5):
        masked_token = "<mask>"
        assert (
            masked_token in masked_input and masked_input.count(masked_token) == 1
        ), "Please add one {0} token for the input, eg: 'He is a {0} guy'".format(
            masked_token
        )

        text_spans = masked_input.split(masked_token)
        text_spans_bpe = (
            (" {0} ".format(masked_token))
            .join([self.bpe.encode(text_span.rstrip()) for text_span in text_spans])
            .strip()
        )
        tokens = self.task.source_dictionary.encode_line(
            "<s> " + text_spans_bpe + " </s>",
            append_eos=False,
            add_if_not_exist=False,
        )

        masked_index = (tokens == self.task.mask_idx).nonzero(as_tuple=False)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        with utils.model_eval(self.model):
            features, extra = self.model(
                tokens.long().to(device=self.device),
                features_only=False,
                return_all_hiddens=False,
            )
        logits = features[0, masked_index, :].squeeze()
        prob = logits.softmax(dim=0)
        values, index = prob.topk(k=topk, dim=0)
        topk_predicted_token_bpe = self.task.source_dictionary.string(index)

        topk_filled_outputs = []
        for index, predicted_token_bpe in enumerate(
            topk_predicted_token_bpe.split(" ")
        ):
            predicted_token = self.bpe.decode(predicted_token_bpe)
            # Quick hack to fix https://github.com/pytorch/fairseq/issues/1306
            if predicted_token_bpe.startswith("\u2581"):
                predicted_token = " " + predicted_token
            if " {0}".format(masked_token) in masked_input:
                topk_filled_outputs.append(
                    (
                        masked_input.replace(
                            " {0}".format(masked_token), predicted_token
                        ),
                        values[index].item(),
                        predicted_token,
                    )
                )
            else:
                topk_filled_outputs.append(
                    (
                        masked_input.replace(masked_token, predicted_token),
                        values[index].item(),
                        predicted_token,
                    )
                )
        return topk_filled_outputs


    def reconstruction_prob_tok(self, masked_tokens, target_tokens, reconstruct=False, source_model=None, topk=1):

        single = False

        # expand batch size 1
        if masked_tokens.dim() == 1:
            single = True
            masked_tokens = masked_tokens.unsqueeze(0)
            target_tokens = target_tokens.unsqueeze(0)

        masked_fill = torch.ones_like(masked_tokens)

        masked_index = (target_tokens != self.task.source_dictionary.pad_index).nonzero(as_tuple=True)
        masked_orig_index = target_tokens[masked_index]

        # edge case of no masked tokens
        if len(masked_orig_index) == 0:
            if reconstruct:
                return masked_tokens, masked_fill, torch.full((len(masked_tokens),), 0.0)
            else:
                return 1.0

        masked_orig_enum = [list(range(len(masked_orig_index))), masked_orig_index]

        with utils.model_eval(self.model):
            features, _ = self.model(
                masked_tokens.long().to(device=self.device),
                features_only=False,
                return_all_hiddens=False,
            )
            probs = features[masked_index]

        if source_model:
            # normalize before subtracting
            probs = logits.softmax(dim=-1)
            orig_probs = probs.detach().clone()
            with utils.model_eval(source_model):
                s_features, _ = source_model.model(
                    masked_tokens.long().to(device=self.device),
                    features_only=False,
                    return_all_hiddens=False,
                )
                s_logits = s_features[masked_index]
                s_probs = s_logits.softmax(dim=-1)
                probs = probs - s_probs
                probs -= torch.min(probs, dim=-1).values.unsqueeze(-1)

        if self.comp_model:
            negate = 0
        else:
            negate = float('-inf')

        for i in range(len(probs)):
            probs[i][self.task.source_dictionary.bos()] = negate # 0
            probs[i][self.task.source_dictionary.eos()] = negate # 2
            probs[i][self.task.source_dictionary.pad()] = negate # 1
            probs[i][-4:] = negate

        if (reconstruct):

            recon_prob = 0

            # sample from topk if not unrestricted
            if topk != -1:
                probs, indices = probs.topk(k=topk, dim=-1)
            # normalize after topk if necessary
            if not source_model:
                probs = probs.softmax(dim=-1)

            if (len(masked_index) > 1):
                tok_samples = [torch.multinomial(prob, 1) for prob in probs]
                if topk != -1:
                    samples = torch.cat([idx[tok] for tok, idx in zip(tok_samples, indices)])
                else:
                    samples = torch.cat(tok_samples)

                (masked_i, masked_j) = masked_index
                recon_prob = []
                idx = 0
                curr_prob = 0
                curr_i = 0
                while idx < len(masked_i):
                    if masked_i[idx] != curr_i:
                        recon_prob.append(curr_prob)
                        curr_prob = 0
                        curr_i += 1
                        continue
                    if source_model:
                        curr_prob += torch.log(orig_probs[idx][tok_samples[idx]])
                    else:
                        curr_prob += torch.log(probs[idx][tok_samples[idx]])
                    idx += 1
                recon_prob.append(curr_prob)
                while len(recon_prob) < len(masked_tokens):
                    recon_prob.append(-float('inf'))
                recon_prob = torch.tensor(recon_prob)

            else:
                tok_sample = torch.multinomial(probs, 1)
                if topk != -1:
                    samples = indices[tok_sample]
                else:
                    samples = tok_sample
                if source_model:
                    recon_prob = torch.tensor([orig_probs[tok_sample].item()])
                else:
                    recon_prob = torch.tensor([probs[tok_sample].item()])

            # set samples
            masked_tokens[masked_index] = samples
            masked_fill[masked_index] = samples

            if single:
                return masked_tokens[0], masked_fill[0], recon_prob[0]
            else:
                return masked_tokens, masked_fill, recon_prob

        return torch.sum(torch.log(probs[masked_orig_enum])).item()

    def sequence_probability(self, tokens):
        single = False

        # expand batch size 1
        if tokens.dim() == 1:
            single = True
            tokens = tokens.unsqueeze(0)

        with utils.eval(self.model):
            features, extra = self.model(
                tokens.long().to(device=self.device),
                features_only=False,
                return_all_hiddens=False,
            )

        tokens = tokens.unsqueeze(-1)
        logits = torch.gather(features, -1, tokens).squeeze(-1)
        probs = logits.softmax(dim=-1)

        if single:
            return probs[0]
        else:
            return probs

    def max_probability(self, tokens):
        single = False

        # expand batch size 1
        if tokens.dim() == 1:
            single = True
            tokens = tokens.unsqueeze(0)

        with utils.eval(self.model):
            features, extra = self.model(
                tokens.long().to(device=self.device),
                features_only=False,
                return_all_hiddens=False,
            )

        logits, indices = torch.max(features, dim=-1)
        probs = logits.softmax(dim=-1)

        if single:
            return probs[0], indices[0]
        else:
            return probs, indices



    def disambiguate_pronoun(self, sentence: str) -> bool:
        """
        Usage::

            >>> disambiguate_pronoun('The _trophy_ would not fit in the brown suitcase because [it] was too big.')
            True

            >>> disambiguate_pronoun('The trophy would not fit in the brown suitcase because [it] was too big.')
            'The trophy'
        """
        assert hasattr(
            self.task, "disambiguate_pronoun"
        ), "roberta.disambiguate_pronoun() requires a model trained with the WSC task."
        with utils.model_eval(self.model):
            return self.task.disambiguate_pronoun(
                self.model, sentence, use_cuda=self.device.type == "cuda"
            )
