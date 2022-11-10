"""
@author: Fabian Schaipp
"""

import numpy as np
import torch
import itertools
import copy

#%% copied from 
"""
Utility functions for Experiments.
"""

def create_exp_group(exp_config: dict):
    """
    This function was adapted from: https://github.com/haven-ai/haven-ai/blob/master/haven/haven_utils/exp_utils.py
    
    Creates a cartesian product of a experiment config.
    
    Each value of exp_config should be a single entry or a list.
    For list values, every entry of the list defines a single realization.
    
    Parameters
    ----------
    exp_config : dict
 
    Returns
    -------
    exp_list: list
        A list of configs, each defining a single run.
    """
    exp_config_copy = copy.deepcopy(exp_config)

    # Make sure each value is a list
    for k, v in exp_config_copy.items():
        if not isinstance(exp_config_copy[k], list):
            exp_config_copy[k] = [v]

    # Create the cartesian product
    exp_list_raw = (
        dict(zip(exp_config_copy.keys(), values)) for values in itertools.product(*exp_config_copy.values())
    )

    # Convert into a list
    exp_list = []
    for exp_dict in exp_list_raw:
        exp_list += [exp_dict]

    return exp_list

#%% 
"""
Utility functions for Pytorch models.
"""

def reset_model_params(model: torch.nn.Module):
    """
    resets all parameters of a Pytorch model
    from: 
        https://discuss.pytorch.org/t/how-to-reset-model-weights-to-effectively-implement-crossvalidation/53859/2
    """
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    return

@torch.no_grad()
def l2_norm(model: torch.nn.Module):
    """
    compute l2 norm of a Pytorch model.
    """
    w = 0.
    for p in model.parameters():
        w += (p**2).sum()
    return torch.sqrt(w).item()

@torch.no_grad()
def grad_norm(opt):
    """
    Computes norm of (stochastic) gradient.
    """
    grad_norm = 0.        
    for group in opt.param_groups:
        for p in group['params']:
            grad_norm += torch.sum(torch.mul(p.grad, p.grad))
     
    return torch.sqrt(grad_norm).item()

