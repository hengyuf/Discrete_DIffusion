#!/usr/bin/env python


import argparse
import numpy       as np
import torch
import os
import json
import zipfile
import urllib.request
from torch.utils.data import Dataset
import torch
import os
import json
import zipfile
import urllib.request
from torch.utils.data import Dataset


DATA_PATH = './datasets'


DATA_PATH = './datasets'


class BanditDataset(Dataset):
    """


    """

    def __init__(self, root=DATA_PATH, seq_len=512, split='train', download=False):
        assert split in {'train', 'valid', 'test'}
        self.root = os.path.join(root, 'bandit')
        self.seq_len = seq_len
        self.split = split

        # Load data
        self.data = self._preprocess_data(split)

        # self.data = torch.load(self.processed_file(split))

    def __getitem__(self, index):
        return self.data[index], self.seq_len

    def __len__(self):
        return len(self.data)

    def _preprocess_data(self, split):
        # Read raw data
        rawdata = torch.load(self.raw_file)
        rawdata = np.array(rawdata)
        

        n_train = int(12800)
        n_valid = int(1600)
        n_test = int(1600)

        # Extract subset
        if split == 'train':
            rawdata = rawdata[:n_train]
        elif split == 'valid':
            rawdata = rawdata[n_train:n_train+n_valid]
        elif split == 'test':
            rawdata = rawdata[n_train+n_valid:n_train+n_valid+n_test]

        data = torch.tensor(rawdata)
        return data

    @property
    def raw_file(self):
        return os.path.join(self.root, 'bandit.npy')

