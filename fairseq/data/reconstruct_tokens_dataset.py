from functools import lru_cache

import numpy as np
import torch

from fairseq.data import data_utils, Dictionary
from fairseq import utils

from . import BaseWrapperDataset, LRUCacheDataset

class ReconstructTokensDataset(BaseWrapperDataset):

    @classmethod
    def apply_reconstruct(
            cls,
            dataset: torch.utils.data.Dataset,
            target_dataset: torch.utils.data.Dataset,
            *args,
            **kwargs
        ):
        """Return the source and target datasets for masked LM training."""
        dataset = LRUCacheDataset(dataset)
        target_dataset = LRUCacheDataset(target_dataset)
        return LRUCacheDataset(cls(dataset,
                                   target_dataset,
                                   *args,
                                   **kwargs))

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        target_dataset: torch.utils.data.Dataset,
        pad_idx: int,
        mask_idx: int,
        bos_idx: int,
        eos_idx: int,
        recon_model,
        comp_model,
        #device,
        seed: int = 1,
        topk: int = -1,
        depth: int = 0,
    ):
        self.dataset = dataset
        self.target_dataset = target_dataset
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.seed = seed
        self.topk = topk
        self.recon_model = recon_model
        if comp_model:
            self.comp_model = comp_model.model
        else:
            self.comp_model = None
        #self.device = device
        self.depth = depth
        self.epoch = 0

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        if hasattr(self.target_dataset, 'set_epoch'):
            self.target_dataset.set_epoch(epoch)
            self.epoch = epoch

    @lru_cache(maxsize=8)
    def __getitem__(self, index: int):
        if not hasattr(self, 'depth'):
            self.depth = 0
        with data_utils.numpy_seed(self.seed, self.epoch, index, self.depth):
            masked_item = self.dataset[index]
            target_item = self.target_dataset[index]

            masked_tokens = masked_item.unsqueeze(0)
            target_tokens = target_item.unsqueeze(0)

            masked_index = (target_tokens != self.pad_idx).nonzero(as_tuple=True)
            masked_orig_index = target_tokens[masked_index]

            # edge case of no masked tokens
            if len(masked_orig_index) == 0:
                return masked_tokens[0]

            masked_orig_enum = [list(range(len(masked_orig_index))), masked_orig_index]

            with utils.model_eval(self.recon_model):
                features, _ = self.recon_model(
                    masked_tokens.long().cuda(),#.to(device=self.device),
                    features_only=False,
                    return_all_hiddens=False,
                )
            probs = features[masked_index]

            if self.comp_model:
                probs = probs.softmax(dim=-1)
                with utils.model_eval(self.comp_model):
                    cfeatures, _ = self.comp_model(
                        masked_tokens.long().cuda(),#.to(device=self.device),
                        features_only=False,
                        return_all_hiddens=False,
                    )
                clogits = cfeatures[masked_index]
                cprobs = clogits.softmax(dim=-1)
                prob_diffs = probs - cprobs
                prob_mask = (prob_diffs > 0).float()
                if not hasattr(self, 'lamb'):
                    self.lamb = 1
                prob_mask[prob_mask != 1] = torch.exp(self.lamb * prob_diffs[prob_mask != 1])
                probs = probs * prob_mask

            if self.comp_model:
                negate = 0
            else:
                negate = float('-inf')

            for i in range(len(probs)):
                probs[i][self.pad_idx] = negate # 0
                probs[i][self.eos_idx] = negate # 2
                probs[i][self.bos_idx] = negate # 1
                probs[i][self.mask_idx] = negate
                probs[i][-4:] = negate # unused tokens

            # sample from topk if not unrestricted
            if self.topk != -1:
                probs, indices = probs.topk(k=self.topk, dim=-1)

            if not self.comp_model:
                probs = probs.softmax(dim=-1)

            if (len(masked_index) > 1):
                tok_samples = [torch.multinomial(prob, 1) for prob in probs]
                if self.topk != -1:
                    samples = torch.cat([idx[tok] for idx, tok in zip(indices, tok_samples)])
                else:
                    samples = torch.cat(tok_samples)

            else:

                tok_sample = torch.multinomial(probs, 1)
                if self.topk != -1:
                    samples = indices[tok_sample]
                else:
                    samples = tok_sample

            # set samples
            masked_tokens[masked_index] = samples.cpu()

            #print("RECONSTRUCTED, depth %d=============" % self.depth)
            #print(masked_tokens[0], flush=True)
            #print("====================================")

            return masked_tokens[0]

