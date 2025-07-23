import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import warnings

from .main import DataClass, SequenceDataClass

def tuple_to_dict_collate(samples):
    batch = default_collate(samples)
    return {"data": batch[0], "targets": batch[1], "ind": batch[2]}

def get_loader(ds: DataClass,
               seed: int=0,
               batch_size: int=1,
               num_workers: int=0,
               drop_last: bool=False
    ):
    
    if isinstance(ds, SequenceDataClass):
        collate_fn = tuple_to_dict_collate
    else:
        collate_fn = None

    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(ds,
                        drop_last=drop_last,
                        shuffle=True,
                        generator=gen,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        collate_fn=collate_fn
    )

    return loader

def infer_shapes(D: DataClass):
    _tmp_dl = get_loader(D,
                         batch_size=1,
    )   
    batch = next(iter(_tmp_dl))
    
    if len(batch['data'].shape) == 1:
        warnings.warn("Inputs have only one dimension (batch dim).")
        input_dim = 0    
    else:
        input_dim = tuple(batch['data'].shape[1:])
        
    if len(batch['targets'].shape) == 1:
        output_dim = (batch['targets'].shape[0],) 
    else:
        output_dim = tuple(batch['targets'].shape[1:])
    
    return input_dim, output_dim