# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from . import BaseWrapperDataset


class SwitchoutDataset(BaseWrapperDataset):

    def switch_out(self, sents, tau, vocab_size,bos_id,eos_id,pad_id):
        mask = torch.eq(sents, bos_id) | torch.eq(sents, eos_id) | torch.eq(sents, pad_id)
        mask = mask.data.type('torch.BoolTensor') #converting to byte tensor for masked_fill in built function
        lengths = (~mask).float().sum(dim=1)
        batch_size, n_steps = sents.size()

        # first, sample the number of words to corrupt for each sentence
        logits = torch.arange(n_steps).type(torch.DoubleTensor)
        logits = logits.mul_(-1).unsqueeze(0).expand_as(sents).contiguous().masked_fill_(mask, float("-inf"))
        probs = torch.nn.functional.softmax(logits.mul_(tau), dim=1)

        # finding corrupt sampels (most likely empty or 1 word) leading to zero prob
        for idx,prob in enumerate(probs.data):
            if torch.sum(prob)<= 0 and idx!=0:
                valid_ind = list(set(range(len(probs.data))))- list(set([idx]))
                for i in range(100):
                    new_indx = random.choice(valid_list)
                    if not torch.sum(probs.data[new_indx])<= 0:
                        probs[idx] = probs[new_indx]
                        break
                    else:
                        pass

        # still num_words probs fails likely due to corrupt input, therefore returning the whole original batch
        try:
            num_words = torch.distributions.Categorical(probs).sample()
        except:
            return sents

        corrupt_pos = num_words.data.float().div_(lengths).unsqueeze(1).expand_as(sents).contiguous().masked_fill_(mask, 0)
        corrupt_pos = torch.bernoulli(corrupt_pos, out=corrupt_pos).type('torch.BoolTensor')
        total_words = int(corrupt_pos.sum())

        # sample the corrupted values, which will be added to sents
        corrupt_val = torch.LongTensor(total_words)
        valid_vocab = list(range(vocab_size))
        valid_vocab.remove(bos_id)
        valid_vocab.remove(eos_id)
        valid_vocab.remove(pad_id)
        corrupt_val = torch.tensor(np.random.choice(valid_vocab, total_words))
        sents[corrupt_pos] = corrupt_val

        return sents

    def __init__(self, dataset, vocab_size, temperature, bos_id, eos_id, pad_id):
        super().__init__(dataset)
        self.temperature = temperature
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item = self.switch_out(item.unsqueeze(0), self.temperature, self.vocab_size, self.bos_id, self.eos_id, self.pad_id).squeeze(0)
        return item
