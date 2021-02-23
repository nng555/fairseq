# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np

from . import FairseqDataset

logger = logging.getLogger(__name__)

class PateTeacherDataset(FairseqDataset):

    def __init__(self, dataset, teacher_idx, max_idx):
        super().__init__()
        assert teacher_idx < max_idx
        self.dataset = dataset
        self.teacher_idx = teacher_idx
        self.max_idx = max_idx
        self.max_samples = int(len(self.dataset)/max_idx) + 1
        start_idx = self.max_samples * self.teacher_idx
        end_idx = min(self.max_samples * (self.teacher_idx + 1), len(self.dataset))
        self.indices = np.arange(len(dataset))[start_idx:end_idx]
        logger.info(
            "selected {} values from size {} dataset (index={})".format(
                end_idx - start_idx,
                len(self.dataset),
                self.teacher_idx,
            )
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def collater(self, samples):
        return self.dataset.collater(samples)

    @property
    def sizes(self):
        return self.dataset.sizes[self.indices]

    def size(self, index):
        return self.dataset.size(self.indices[index])

    @property
    def name(self):
        return self.dataset.name

    def num_tokens(self, index):
        return self.dataset.num_tokens(self.indices[index])

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)

    def prefetch(self, indices):
        self.dataset.prefetch(self.indices[indices])
