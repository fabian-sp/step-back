import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

from .cifar import get_cifar10, get_cifar100
from .mnist import get_mnist
from .synthetic import get_synthetic_matrix_fac, get_synthetic_linear
from .libsvm import LIBSVM_NAMES, get_libsvm
from .sensor import get_sensor
from .imagenet32 import get_imagenet32
from .imagenet import get_imagenet
from .shakespeare import get_shakespeare

# Custom Dataset handler
# main difference: also returns the index of a batch in __getitem__
# we return a dict (not standard). sometimes this needs to be done later with collate_fn in DataLoader 
class DataClass:
    def __init__(self, dataset: Dataset, split: str):
        self.dataset = dataset
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, targets = self.dataset[index]

        return {'data': data, 
                'targets': targets, 
                'ind': index}

# Dataset handler for sequential datasets (language)
# adapted from: github.com/epfml/llm-baselines/blob/main/src/data/utils.py
# NOTE: we map from a tuple to dict with a custom collate_fn in the DataLoader
class SequenceDataClass(DataClass):
    def __init__(self, dataset, sequence_length: int, split: str):
        super().__init__(dataset=dataset, split=split)
        self.sequence_length = sequence_length

    def __len__(self):
        total_length = len(self.dataset)
        # chunk the data into sequences of length `sequence_length`
        # NOTE: we discard the last remainding sequence if it's not of length `sequence_length`
        return (total_length - 1) // self.sequence_length

    def __getitem__(self, index):
        seq_length = self.sequence_length
        index = index * seq_length
        x = torch.from_numpy((self.dataset[index : index + seq_length]).astype(np.int64))

        y = torch.from_numpy(
            (self.dataset[index + 1 : index + 1 + seq_length]).astype(np.int64)
        )
        return x, y, index
    
def get_dataset(config: dict, split: str, seed: int, path: str) -> DataClass:
    """
    Main function mapping a dataset name to an instance of DataClass.   
    """
    
    assert split in ['train', 'val']
    
    kwargs = config['dataset_kwargs']
    name = config['dataset']
    
    #==== all dataset options ======================
    if name == 'cifar10':
        ds = get_cifar10(split=split, path=path)
    
    elif name == 'cifar100':
        ds = get_cifar100(split=split, path=path)
    
    elif name == 'mnist':
        ds = get_mnist(split=split, path=path, **kwargs)
        
    elif name == 'synthetic_matrix_fac':
        assert all([k in kwargs.keys() for k in ['p', 'q', 'n_samples']]), "For synthetic dataset, dimensions and number of samples need to be specified in config['dataset_kwargs]']."
        ds = get_synthetic_matrix_fac(split=split, seed=seed,
                                      **kwargs)
    elif name == 'sensor':
        ds = get_sensor(split=split, path=path)

    elif name == 'synthetic_linear':
        assert all([k in kwargs.keys() for k in ['p', 'n_samples']]), "For synthetic dataset, dimensions and number of samples need to be specified in config['dataset_kwargs]']."
        
        if config['loss_func'] in ['logistic']:
            classify = True
        else:
            classify = False
    
        ds = get_synthetic_linear(classify=classify, split=split, seed=seed, **kwargs)
    
    elif name in LIBSVM_NAMES:
        ds = get_libsvm(split=split, name=name, path=path, **kwargs)

    elif name == 'imagenet32':
        ds = get_imagenet32(split=split, path=path)
    
    elif name == 'imagenet':
        ds = get_imagenet(split=split, path=path)
    
    elif name == 'shakespeare':
        ds = get_shakespeare(split=split, path=path)

    else:
        raise KeyError(f"Unknown dataset name {name}.")
    #===============================================
    
    # here come datasets that need a sequential __getitem__
    if name in ["shakespeare"]:
        sequence_length = config["model_kwargs"]["seq_len"]
        D = SequenceDataClass(dataset=ds, sequence_length=sequence_length, split=split)
    else:
        D = DataClass(dataset=ds, split=split)
    
    return D
    
