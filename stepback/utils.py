"""
@author: Fabian Schaipp
"""

import numpy as np
import torch
import itertools
import copy
import os

from sklearn.linear_model import Ridge, LogisticRegression

#%%
"""
Utility functions for managing output files.
"""

def get_output_filenames(exp_id, output_dir='output/'):
    """Given exp_id, it returns list of output filenames.
    The output filename should either be identical to exp_id or have the form <exp_id>-1, <exp_id>-2 etc
    """
    all_files = os.listdir(output_dir)
    exp_files = [f for f in all_files if (f == exp_id+'.json') or (exp_id+'-' in f)] 

    return exp_files
#%%
"""
Utility functions for Experiments.
"""

def prepare_config(exp_config: dict) -> dict:
    """
    Given an experiment config, we do the following preparations:
        
        * Convert n_runs to a list of run_id (integer values)
        * Convert each element of opt to a list of opt configs.
    """
    c = copy.deepcopy(exp_config)
    
    c['run_id'] = list(range(c['n_runs']))
    del c['n_runs']
    
    
    assert isinstance(c['opt'], list), f"The value of 'opt' needs to be a list, but is given as {c['opt']}."
    
    all_opt = list()
    for this_opt in c['opt']:
        
        # make every value a list
        for k in this_opt.keys():
            if not isinstance(this_opt[k], list):
                this_opt[k] = [this_opt[k]]
        
        # cartesian product
        all_opt += [dict(zip(this_opt.keys(), v)) for v in itertools.product(*this_opt.values())]
         
    c['opt'] = all_opt
    
    return c

def create_exp_list(exp_config: dict):
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
        dict(zip(exp_config_copy.keys(), v)) for v in itertools.product(*exp_config_copy.values())
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
def grad_norm(model: torch.nn.Module):
    """
    Computes norm of (stochastic) gradient.
    """
    g = 0.        
    for p in model.parameters():
        g += torch.sum(torch.mul(p.grad, p.grad))
     
    return torch.sqrt(g).item()

#%%
"""
Compute optimal value for convex problems.
"""

def ridge_opt_value(X, y, lmbda, fit_intercept=False):
    n_samples = len(y)
    sk = Ridge(alpha=n_samples * lmbda/2, fit_intercept=fit_intercept, tol=1e-10, solver='auto', random_state=1234)
    
    sk.fit(X,y)
    sol = sk.coef_
    
    if fit_intercept:
        t2 = ((X@sol + sk.intercept_ - y)**2).mean()
    else:
        t2 = ((X@sol - y)**2).mean()
        
    t1 = lmbda/2 * np.linalg.norm(sol)**2
    
    return t1+t2

def logreg_opt_value(X, y, lmbda, fit_intercept=False):
    n_samples = len(y)
    
    if lmbda > 0:
        sk = LogisticRegression(C=1/(n_samples*lmbda), penalty='l2', fit_intercept=False, tol=1e-10, random_state=1234)
    else:
        sk = LogisticRegression(penalty='none', fit_intercept=False, tol=1e-10, random_state=1234)
        
    sk.fit(X,y)
    sol = sk.coef_.squeeze()
    print(sol)
    
    if fit_intercept:
        t2 = np.log(1+np.exp(-y*(X@sol + sk.intercept_))).mean()
    else:
        t2 = np.log(1+np.exp(-y*(X@sol))).mean()
        
    t1 = lmbda/2 * np.linalg.norm(sol)**2
    return t1+t2
