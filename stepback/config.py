import copy
import json
import os
import itertools

from .defaults import DEFAULTS

class ConfigManager:
    """
    For managing config files.

    We distinguish two types of config files:
        * dict-type, where each value can be a list. This will be converted into a cross-product of single run configs.
            You should always set up this type of config, and then create list-type configs only for temporary use.

        * list-type. This is essentially a subset of the cross-product that comes from a dict-type config. Intenden mainly for running many jobs in parallel.

    """
    def __init__(self, 
                 exp_id: str, 
                 config_dir: str=DEFAULTS.config_dir
                 ):

        self.exp_id = exp_id
        self.config_dir = config_dir


    def create_config_list(self):
        """
        Creates a list of all configs for single runs.

        Operation depends on which type of config the JSON with name ``self.exp_id`` is:
            
            * If dict-type, then the cross-product is created here and returned.
            * If list-type, then the list is returned.

        """

        with open(os.path.join(self.config_dir, self.exp_id) + '.json') as f:
            exp_config = json.load(f)
        
        # Check whether it is dict-typ or not, and do some sanity checks
        if isinstance(exp_config, dict):
            self.dict_type = True
            assert 'n_runs' in exp_config.keys(), 'Dict-type config must specify the number of runs (e.g. "n_runs": 1).'
        elif isinstance(exp_config, list):
            self.dict_type = False
            for c in exp_config:
                assert 'run_id' in c.keys(), 'List-type config must contain "run_id" for every list element.'
        else:
            raise KeyError("Config has unknown format, must be dict or list.")

        if self.dict_type:
            exp_config = prepare_config(exp_config)
            exp_list = create_exp_list(exp_config)          # cartesian product
        else:
            exp_list = copy.deepcopy(exp_config)

        self.exp_list = exp_list

        return self.exp_list

        
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