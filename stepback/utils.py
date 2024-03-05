"""
@author: Fabian Schaipp
"""

import numpy as np
import torch
import itertools
import copy
import os
import warnings
import json

from sklearn.linear_model import Ridge, LogisticRegression

from .log import Container
from .config import ConfigManager

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
        
        # if stored as single dict ouput, make list of length one
        if isinstance(C.data, dict):
            for key in ["config", "summary", "history"]:
                assert key in C.data.keys()
            C.data = [C.data]

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
Utility functions for Config files.
"""

def split_config(exp_id: str, job_name: str, config_dir: str, splits: int=None, only_new: bool=False, output_dir: str='output/'):
    """Splits a dict-type config into parts.

    Parameters
    ----------
    exp_id : str
        The name of the dict-type config.
    job_name : str
        Folder name where the temporary config files will be stored.
    config_dir : str
        Directory where ``exp_id.json```is stored. Temporary configs will be created in this directory as well.
    splits : int, optional
        How many parts (of roughtly equal size) you want to split, by default None.
        If not specified, then one single config per file.
    only_new : bool, optional
        Whether to only create configs which have not been run, by default False.

        Use this option with caution. We will look up all files that start with ``exp_id-`` (or are ``exp_id``) in the output directory specified.
        Any config that can be found in those files will be disregarded.
    output_dir : str, optional
        Directory of output files, by default 'output'.
        Only relevant if ``only_new=True``.
    """

    if os.path.exists(os.path.join(config_dir, job_name)):
        warnings.warn("A folder with the same job__name already exists, files will be overwritten.")
    else:
        os.mkdir(os.path.join(config_dir, job_name))

    # Load config_list
    Conf = ConfigManager(exp_id=exp_id, config_dir=config_dir)
    config_list = Conf.create_config_list()
    assert Conf.dict_type, "For splitting a config, it should be of dict-type"
    print(f"Initial config contains {len(config_list)} elements.")

    # If only new, load output files and keep only the ones that have not been run yet
    # Check all output files which start with 'exp_id-'
    if only_new:
        print("Screening for existing runs...")
        existing_files = get_output_filenames(exp_id, output_dir=output_dir)
        existing_configs = list()

        for _e in existing_files:
            print(f"Looking in output data from {output_dir+_e}")
            C = Container(name=_e, output_dir=output_dir, as_json=True)
            C.load() # load data
            existing_configs += [copy.deepcopy(_d['config']) for _d in C.data]
            del C

        to_remove = list()
        for _conf in config_list:
            
            # Base adds empty kwargs if not specified; we do this here to ensure correct comparison
            if 'dataset_kwargs' not in _conf.keys():
                _conf['dataset_kwargs'] = dict()

            if 'model_kwargs' not in _conf.keys():
                _conf['model_kwargs'] = dict()

            # Check if exists --> remove
            if _conf in existing_configs:
                to_remove.append(_conf)
            else:
                pass

        config_list = [_conf for _conf in config_list if _conf not in to_remove]        # remove all existing runs
        print(f"Screening ended: {len(config_list)} elements remaining.")

    # Split config_list evenly
    if splits is None:
        splits = len(config_list)

    list_of_config_lists = [list(a) for a in np.array_split(config_list, splits)]

    # store
    for j, _this_list in enumerate(list_of_config_lists):
        with open(os.path.join(config_dir, job_name, exp_id) + f'-{j:02d}.json', "w") as f:
            json.dump(_this_list, f, indent=4, sort_keys=True)

    return

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
    
    # Caution: in scikit-learn, the intercept is not penalized, but in the Pytorch default layout it is.
    # To account for this, we add a column of ones, and penalize the intercept.
    if fit_intercept:
        X = np.hstack((X, np.ones((n_samples,1))))

    # We account for intercept with column of ones, see above
    sk = Ridge(alpha=n_samples * lmbda/2, fit_intercept=False, tol=1e-10, solver='auto', random_state=1234)
        
    sk.fit(X,y)
    sol = sk.coef_
    
    t2 = ((X@sol - y)**2).mean()    
    t1 = lmbda/2 * np.linalg.norm(sol)**2
    
    return t1+t2

def logreg_opt_value(X, y, lmbda, fit_intercept=False):
    n_samples = len(y)
    
    # Caution: in scikit-learn, the intercept is not penalized, but in the Pytorch default layout it is.
    # To account for this, we add a column of ones, and penalize the intercept.
    if fit_intercept:
        X = np.hstack((X, np.ones((n_samples,1))))

    # We account for intercept with column of ones, see above
    if lmbda > 0:
        sk = LogisticRegression(C=1/(n_samples*lmbda), penalty='l2', fit_intercept=False, tol=1e-10, random_state=1234)
    else:
        sk = LogisticRegression(penalty=None, fit_intercept=False, tol=1e-10, random_state=1234)
        
    sk.fit(X,y)
    sol = sk.coef_.squeeze()
    
    t2 = np.log(1+np.exp(-y*(X@sol))).mean()    
    t1 = lmbda/2 * np.linalg.norm(sol)**2

    return t1+t2
