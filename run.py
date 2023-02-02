"""
Main file for running experiments.
"""
import json
import copy
from itertools import product

from stepback.utils import prepare_config, create_exp_list
from stepback.base import Base

from stepback.log import Container


CONFIG_DIR = 'configs/'
OUTPUT_DIR = 'output/'

def run_one(exp_id, device='cuda'):
    
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
        B = Base(name=exp_id + f'_{j}',
                config=config, 
                device=device, 
                data_dir='data/')
        
        B.setup()
        B.run() # train and validate
        
        C.append(B.results).store() # store results
    
    return 

if __name__ == '__main__':
    exp_id = 'test1' # file name of config
    run_one(exp_id)
    

