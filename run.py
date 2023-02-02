"""
Main file for running experiments.
"""
import json
from itertools import product
import argparse

from stepback.utils import prepare_config, create_exp_list
from stepback.base import Base

from stepback.log import Container


parser = argparse.ArgumentParser(description='Run stepback from the terminal.')
parser.add_argument('-i', '--id', nargs='?', type=str, default='test1', help="The id of the config (its file name).")
parser.add_argument('-cdir', '--config_dir', nargs='?', type=str, default='configs/', help="The config directory.")
parser.add_argument('-odir', '--output_dir', nargs='?', type=str, default='output/', help="The output directory.")

# for running from IPython
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
    args = parser.parse_args()

    CONFIG_DIR = args.config_dir
    OUTPUT_DIR = args.output_dir
    EXP_ID = args.id

    run_one(EXP_ID)
    

