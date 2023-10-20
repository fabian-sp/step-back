"""
@author: Fabian Schaipp
"""

import numpy as np
import torch
import itertools
import copy
import os

from sklearn.linear_model import Ridge, LogisticRegression

from .log import Container

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

    # sorting
    exp_files.sort() # first sort integers
    exp_files.sort(key=len) # no integer file first

    # remove .json from string
    exp_files = [f.split('.json')[0] for f in exp_files]

    return exp_files

def filter_output_file(exp_id, exp_filter=dict(), opt_filter=dict(), drop_keys=list(), fname=None, output_dir='output/', as_json=True):
    """Filter ouput file. Deletes all results that are specified by exp_filter and opt_filter.
        Values of exp_filter and opt_filter can also be a list to specifiy multiple values. 
        Deletes also all drop_keys in history.
    Example:
        opt_filter = {'name': 'adam'}     # deletes all adam runs
        exp_filter = {'dataset': 'mnist'} # deletes all runs where mnist was the dataset 
        drop_keys = ['step_size_list']     # deletes step sizes 

    CAUTION: This will overwrite the file if not fname is specified. Otherwise, it will write the filtered output in fname.
    """
    print(f"Reading from {os.path.join(output_dir,exp_id)}.")
    C = Container(name=exp_id, output_dir=output_dir, as_json=as_json)
    C.load() # load data
    print(f"Original file has {len(C.data)} entries.")
    new_data = list()

    for d in C.data:
        DROP = False
        conf = copy.deepcopy(d['config'])

        for k,v in exp_filter.items():
            if not isinstance(v, list):
                v = [v]
            if conf.get(k) in v:
                DROP = True
        
        for k,v in opt_filter.items():
            if not isinstance(v, list):
                v = [v]
            
            if conf['opt'].get(k) in v:
                DROP = True

        if not DROP:
            if len(drop_keys) > 0:
                for epoch_rec in d['history']:
                    for dkey in drop_keys:
                        epoch_rec.pop(dkey, None) # drop if exists

            new_data.append(d)  
        else:
            print("Dropping the config:") 
            print(conf)

    NEW_FNAME = exp_id if fname is None else fname
    
    new = Container(name=NEW_FNAME, output_dir=output_dir, as_json=as_json)
    new.data = new_data
 
    print(f"New file has {len(new_data)} entries.")  
    new.store()     

    return

def merge_subfolder(folder_name, fname='merged', output_dir='output/'):
    """Merges all output files from output_dir/folder_name """
    dir = os.path.join(output_dir, folder_name)
    
    if not os.path.exists(dir):
        raise OSError("Folder directory does not exist.")
    
    all_files = os.listdir(dir)
    all_files = [f[:-5] for f in all_files if f[-5:]=='.json']

    # save new file in parent directory of subfolder
    merge_output_files(all_files, fname, output_dir=dir, merged_dir=output_dir, as_json=True)

    return


def merge_output_files(exp_id_list, fname, output_dir='output/', merged_dir=None, as_json=True):
    """ Merges output files from a list of exp_id into a new file (with name fname)."""

    if merged_dir is None:
        merged_dir = output_dir

    merged = Container(name=fname, output_dir=merged_dir, as_json=as_json)

    for _e in exp_id_list:
        C = Container(name=_e, output_dir=output_dir, as_json=as_json)
        C.load() # load data
        merged.data += C.data # append

    all_model = set([d['config']['model'] for d in merged.data])
    assert len(all_model) == 1, f"Found multiple models: {all_model}. Please make sure not to merge results from different setups."
    
    all_dataset = set([d['config']['dataset'] for d in merged.data])
    assert len(all_dataset) == 1, f"Found multiple datasets: {all_dataset}. Please make sure not to merge results from different setups."
    
    print(f"New output file has {len(merged.data)} entries.")
    merged.store()

    return

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
    
    if fit_intercept:
        t2 = np.log(1+np.exp(-y*(X@sol + sk.intercept_))).mean()
    else:
        t2 = np.log(1+np.exp(-y*(X@sol))).mean()
        
    t1 = lmbda/2 * np.linalg.norm(sol)**2
    return t1+t2
