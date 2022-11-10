import torch
from torch import nn

from .basic_models import MLP

def get_model(config: dict={}) -> torch.nn.Module:
    """
    Main function mapping a model name to an instance of torch.nn.Module.  
    """
    kwargs = config['model_kwargs']
    name = config['model']
    
    if name == "linear":
        # here, input_dim should be one-dim
        assert len(config['_input_dim']) == 1, "For linear models, we expect input dimensionality of length 1."
        model = MLP(input_size=config['_input_dim'][0], output_size=1, hidden_sizes=[], bias=False, **kwargs)
    else:
        raise KeyError(f"Unknown model option {name}.")   
    
    return model
        
        