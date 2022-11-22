"""
Main file for running experiments.
"""
import json
import copy
from itertools import product

from src.utils import prepare_config, create_exp_list
from src.base import Base

from src.log import Container


CONFIG_DIR = 'configs/'
OUTPUT_DIR = 'output/'

exp_id = 'test1' # file name of config


def run_one(exp_id):
    
    # load config
    with open(CONFIG_DIR + f'{exp_id}.json') as f:
        exp_config = json.load(f)
    
    # prepare list of configs (cartesian product)
    exp_config = prepare_config(exp_config)
    exp_list = create_exp_list(exp_config)
        
    print(f"Created {len(exp_list)} different configurations.")
    
    # initialize container for storing
    C = Container(name=exp_id, output_dir=OUTPUT_DIR, as_json=True)
    
    for j, config in enumerate(exp_list): 
        # each run gets id, by position in the list
        B = Base(name=exp_id + '_j', config=config, device='cuda', data_dir='data/')
        B.setup()
        B.run() # train and validate
        
        C.append(B.results).store() # store results
    
    return 

run_one(exp_id)
    

