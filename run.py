"""
Main file for running experiments.
"""
import json
from typing import Union
import argparse
import torch

from stepback.utils import prepare_config, create_exp_list
from stepback.base import Base
from stepback.log import Container

from stepback.defaults import DEFAULTS

parser = argparse.ArgumentParser(description='Run stepback from the terminal.')
parser.add_argument('-i', '--id', nargs='?', type=str, default='test1', help="The id of the config (its file name).")
parser.add_argument('-cdir', '--config_dir', nargs='?', type=str, default=DEFAULTS.config_dir, help="The config directory.")
parser.add_argument('-odir', '--output_dir', nargs='?', type=str, default=DEFAULTS.output_dir, help="The output directory.")
parser.add_argument('-ddir', '--data_dir', nargs='?', type=str, default=DEFAULTS.data_dir, help="The data directory.")
parser.add_argument('--device', nargs='?', type=str, default=DEFAULTS.device, help="Device to run on.")

parser.add_argument('-nw', '--num_workers', nargs='?', type=int, default=DEFAULTS.num_workers, help="Number of workers for DataLoader.")
parser.add_argument('--data_parallel', nargs='+', default=DEFAULTS.data_parallel, help='Device list for DataParallel in Pytorch.')
parser.add_argument('--verbose', action="store_true", help="Verbose mode.")
parser.add_argument('--force_deterministic', action="store_true", help="Use deterministic mode in Pytorch. Might require setting environment variables.")

def run_one(exp_id: str,
            config_dir: str=DEFAULTS.config_dir, 
            output_dir: str=DEFAULTS.output_dir, 
            data_dir: str=DEFAULTS.data_dir, 
            device: str=DEFAULTS.device, 
            num_workers: int=DEFAULTS.num_workers,
            data_parallel: Union[list, None]=DEFAULTS.data_parallel,
            verbose: bool=DEFAULTS.verbose,
            force_deterministic: bool=DEFAULTS.force_deterministic
            ):
    
    # load config
    with open(config_dir + f'{exp_id}.json') as f:
        exp_config = json.load(f)
    
    # prepare list of configs (cartesian product)
    exp_config = prepare_config(exp_config)
    exp_list = create_exp_list(exp_config)
        
    print(f"Created {len(exp_list)} different configurations.")
    
    # initialize container for storing
    C = Container(name=exp_id, output_dir=output_dir, as_json=True)
    
    if force_deterministic:
        torch.use_deterministic_algorithms(True)
        print("Using Pytorch deterministic mode. This might lead to substantial slowdown.")

    for j, config in enumerate(exp_list): 
        # each run gets id, by position in the list
        B = Base(name=exp_id + f'_{j}',
                 config=config, 
                 device=device,
                 data_dir=data_dir,
                 num_workers=num_workers,
                 data_parallel=data_parallel,
                 verbose=verbose)
        
        B.setup()
        B.run() # train and validate
        
        C.append(B.results).store() # store results
    
    print("All experiments have completed.")
    
    return 

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    run_one(args.id,
            config_dir=args.config_dir,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            device=args.device,
            num_workers=args.num_workers,
            data_parallel=args.data_parallel,
            verbose=args.verbose,
            force_deterministic=args.force_deterministic)
    

