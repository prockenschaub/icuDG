# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import sys
import inspect

import numpy as np
import torch
from collections import Counter
from itertools import cycle


def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights


def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


def print_separator():
    print("="*80)


def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)


def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)


def slice_to(obj, ind):
    if torch.is_tensor(obj) or isinstance(obj, (list, tuple, np.ndarray)):
        return obj[:ind]
    elif isinstance(obj, dict):
        return {i: obj[i][:ind] for i in obj}    
    else:
        raise ValueError()


def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(yi), len(yj))

        pairs.append(((slice_to(xi, min_n), yi[:min_n]), (slice_to(xj, min_n), yj[:min_n])))

    return pairs


def split_meta_train_test(minibatches, num_meta_test=1):
    n_domains = len(minibatches)
    perm = torch.randperm(n_domains).tolist()
    pairs = []
    meta_train = perm[:(n_domains-num_meta_test)]
    meta_test = perm[-num_meta_test:]

    for i,j in zip(meta_train, cycle(meta_test)):
         xi, yi = minibatches[i][0], minibatches[i][1]
         xj, yj = minibatches[j][0], minibatches[j][1]

         min_n = min(len(xi), len(xj))
         pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def to_device(obj, device):
    if torch.is_tensor(obj) or isinstance(obj, torch.nn.Module):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        return [to_device(i, device) for i in obj]
    elif isinstance(obj, dict):
        return {a: to_device(b, device) if torch.is_tensor(b) or isinstance(b, torch.nn.Module) else b for a,b in obj.items() }
    else:
        raise ValueError("invalid object type passed to to_device")


def list_classes(module):
    classes = inspect.getmembers(module, inspect.isclass)
    names = list(dict(classes).keys())
    return names


def cat(lst):
    # Is this a list of tensors?
    if np.all([torch.is_tensor(i) for i in lst]):
        return torch.cat(lst)

    # Is this a list of lists/tuples?
    elif np.all([isinstance(i, (list, tuple)) for i in lst]):
        return [cat(x) for x in zip(*lst)]       

    # Is this a list of dictionaries?
    elif np.all([isinstance(i, dict) for i in lst]):
        return {k: cat([i[k] for i in lst]) for k in lst[0].keys() }

    raise ValueError(
        f'Can only concatenate lists of tensors, lists, or dicts. '
        f'Got {np.unique([str(i.__class__) for i in lst])}.'
    )


def predict_on_set(algorithm, loader, device, aux_fn=None):
    all_preds, all_targets, all_aux = [], [], []
    with torch.no_grad():
        for batch in loader:
            # Batch: (x, y, ...)
            
            # Make predictions
            x = to_device(batch[0], device)
            
            algorithm.eval()
            preds = algorithm.predict(x).cpu()

            # Extract auxilliary information 
            aux = None
            if aux_fn is not None:
                aux = aux_fn(batch)

            # Store all necessary information
            all_preds += [preds]
            all_targets += [batch[1]]
            all_aux += [aux]
    
    return cat(all_preds), cat(all_targets), all_aux


def add_prefix(dictionary, prefix):
    return {f'{prefix}_{str(key)}': val for key, val in dictionary.items()}


class MovingAverage:
    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                # correction by 1/(1 - self.ema)
                # so that the gradients amplitude backpropagated in data is independent of self.ema
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data


def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()


def pad_to_len(x: np.ndarray, length: int, value: float = 0.) -> np.ndarray:
    """Pad the first dimension of an array to a given length using PAD_VALUE

    Args:
        x (np.ndarray): 2-dimensional array
        length (int): maximum length to pad. If less than `x.shape[0]`, x is shortened.
        value (float): value used as padding

    Raises:
        ValueError: if `x` does not have exactly 2 dimensions

    Returns:
        np.ndarray: padded array
    """
    if not len(x.shape) == 2:
        raise ValueError(f'Only 2-dimensional arrays can be padded, got {len(x.shape)} dims.')
    copy_len = min(x.shape[0], length)
    x_pad = np.full((length, x.shape[1]), value, dtype=np.float32)
    x_pad[:copy_len, :] = x[:copy_len, :]
    return x_pad


def pad_missing(x: np.ndarray, value: float = 0.) -> np.ndarray:
    """Pad missing values using PAD_VALUE

    Args:
        x (np.ndarray)): numpy array with missing values
        value (float): value used as padding

    Returns:
        np.ndarray: padded array
    """
    x[np.isnan(x)] = value
    return x

