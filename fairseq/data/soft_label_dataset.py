# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from . import FairseqDataset


class SoftLabelDataset(FairseqDataset):

    def __init__(self, labels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.labels = labels

    def __getitem__(self, index):
        return self.labels[index]

    def __len__(self):
        return len(self.labels)

    def collater(self, samples):
        return torch.tensor(samples)

