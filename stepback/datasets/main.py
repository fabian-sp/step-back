import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


from .cifar import get_cifar10, get_cifar100
from .mnist import get_mnist
from .synthetic import get_synthetic_matrix_fac, get_synthetic_linear
from .libsvm import LIBSVM_NAME_MAP, get_libsvm
from .sensor import get_sensor
from .imagenet32 import get_imagenet32
from .imagenet import get_imagenet

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
    
    elif name in LIBSVM_NAME_MAP.keys():
        ds = get_libsvm(split=split, name=name, path=path, **kwargs)

    elif name == 'imagenet32':
        ds = get_imagenet32(split=split, path=path)
    
    elif name == 'imagenet':
        ds = get_imagenet(split=split, path=path)

    else:
        raise KeyError(f"Unknown dataset name {name}.")
    #===============================================
    
    D = DataClass(dataset=ds, split=split)
    
    return D

def infer_shapes(D: DataClass):
    _tmp_dl = DataLoader(D, batch_size=1)   
    batch = next(iter(_tmp_dl))
    
    if len(batch['data'].shape) == 1:
        input_dim = (batch['data'].shape[0],)    
    else:
        input_dim = tuple(batch['data'].shape[1:])
        
    if len(batch['targets'].shape) == 1:
        output_dim = (batch['targets'].shape[0],)    
    else:
        output_dim = tuple(batch['targets'].shape[1:])
    
    return input_dim, output_dim
    
