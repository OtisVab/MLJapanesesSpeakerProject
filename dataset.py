import math
import random
import numpy as np
import torch

from torch.utils.data import Dataset as TorchDataset, DataLoader
from sklearn.model_selection import KFold

def collate_sequence(batch):

    # Batch is batch_size x time_series x input_size

    inputs, targets = zip(*batch)

    # Get series lengths
    lengths = torch.LongTensor(list(map(lambda x : x.shape[0], inputs)))

    # Get maximum series length in batch
    max_len = torch.max(lengths).item()

    # Pad each input in the time series
    inputs = torch.stack(list(map(
        lambda x : torch.cat(
            [x, torch.zeros(max_len-x.shape[0], *x.shape[1:])],
            dim=0,
        ),
        inputs,
    )))

    # Return batch_size x max_len x input_size
    return (inputs, lengths), torch.stack(targets)

class SequenceDataLoader(DataLoader):

    def __init__(self, *args, collate_fn=None, **kwargs):
        super().__init__(*args, collate_fn=collate_sequence, **kwargs)

class SpeechData(TorchDataset):

    def __init__(self, filename, block_size, num_LPC_vals=12):

        # X is num_examples x time_steps x LPC_vals, target is num_examples x 1
        self._X, self._t = [], []

        with open(filename, 'r') as f:

            LPC_vals = []
            block_counter = 0
            speaker_id = 0

            for line in f.readlines():

                if line == '\n' or line=='':
                    continue

                line_vals = np.array(
                    list(map(
                        lambda x : float(x),
                        line.split(' ')[:num_LPC_vals]
                    ))
                )

                # If the current line has LPC vals, append them and move to the next line
                if not np.isclose(line_vals, 1.0).all():
                    LPC_vals.append(line_vals)
                    continue    

                # (else)

                # Append current input
                self._X.append(
                    torch.Tensor(np.stack(LPC_vals))
                )

                # Append current target
                self._t.append(
                    torch.tensor(speaker_id).long()
                )

                block_counter += 1 
                LPC_vals = []

                if isinstance(block_size, int):
                    speaker_id = block_counter//block_size
                else:
                    if block_counter >= block_size[speaker_id]:
                        block_counter = 0
                        speaker_id += 1

        rand = list(zip(self._X, self._t))
        random.shuffle(rand)
        self._X, self._t = zip(*rand)

        self._X, self._t = np.array(self._X, dtype=object), np.array(self._t, dtype=object)

    def __getitem__(self, item):
        return self._X[item], self._t[item]

    def __len__(self):
        return len(self._t)

    @property
    def X(self):
        return self._X

    @property
    def t(self):
        return self._t

class DataLoaderWrapper(SpeechData):

    def __init__(self, data, target):

        rand = list(zip(data, target))
        random.shuffle(rand)
        self._X, self._t = zip(*rand)

        self._X, self._t = np.array(self._X, dtype=object), np.array(self._t, dtype=object)

    def __getitem__(self, item):
        return self._X[item], self._t[item]

    def __len__(self):
        return len(self._t)


def SplitDataLoader(dataloader, n_folds, shuffle=True):
    
    kf = KFold(n_splits=n_folds, shuffle=shuffle)
    for train_idx, test_idx in kf.split(dataloader.X):
        yield DataLoaderWrapper(dataloader.X[train_idx], dataloader.t[train_idx]), DataLoaderWrapper(dataloader.X[test_idx], dataloader.t[test_idx])