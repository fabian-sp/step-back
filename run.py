"""
Main file for running experiments.
"""
import json
from typing import Union
import argparse
import torch

from stepback.base import Base
from stepback.log import Container
from stepback.config import ConfigManager

from stepback.defaults import DEFAULTS

parser = argparse.ArgumentParser(description='Run stepback from the terminal.')
parser.add_argument('-i', '--id', nargs='?', type=str, default='test1', help="The id of the config (its file name).")
parser.add_argument('-cdir', '--config-dir', nargs='?', type=str, default=DEFAULTS.config_dir, help="The config directory.")
parser.add_argument('-odir', '--output-dir', nargs='?', type=str, default=DEFAULTS.output_dir, help="The output directory.")
parser.add_argument('-ddir', '--data-dir', nargs='?', type=str, default=DEFAULTS.data_dir, help="The data directory.")
parser.add_argument('--device', nargs='?', type=str, default=DEFAULTS.device, help="Device to run on.")

parser.add_argument('-nw', '--num-workers', nargs='?', type=int, default=DEFAULTS.num_workers, help="Number of workers for DataLoader.")
parser.add_argument('--data-parallel', nargs='+', default=DEFAULTS.data_parallel, help='Device list for DataParallel in Pytorch.')
parser.add_argument('-logk', '--log-every-k-steps', nargs='?', type=int, default=DEFAULTS.log_every_k_steps, help="Stepwise logging.")
parser.add_argument('--verbose', action="store_true", help="Verbose mode.")
parser.add_argument('--force-deterministic', action="store_true", help="Use deterministic mode in Pytorch. Might require setting environment variables.")

def run_one(exp_id: str,
            config_dir: str=DEFAULTS.config_dir, 
            output_dir: str=DEFAULTS.output_dir, 
            data_dir: str=DEFAULTS.data_dir, 
            device: str=DEFAULTS.device, 
            num_workers: int=DEFAULTS.num_workers,
            data_parallel: Union[list, None]=DEFAULTS.data_parallel,
            log_every_k_steps: Union[int, None]=DEFAULTS.log_every_k_steps,
            verbose: bool=DEFAULTS.verbose,
            force_deterministic: bool=DEFAULTS.force_deterministic
            ):
    """Function for running all runs from one config file.
    Default values for all arguments can be found in ``stepback/defaults.py``.

    Parameters
    ----------
    exp_id : str
        The experiment ID, equal to the name of the config file.
    config_dir : str, optional
        Directory where config file is stored, by default DEFAULTS.config_dir
    output_dir : str, optional
        Directory where output is stored, by default DEFAULTS.output_dir
    data_dir : str, optional
        Directory where datasets can be found,, by default DEFAULTS.data_dir
    device : str, optional
        Device string, by default DEFAULTS.device
        If 'cuda' is specified, but not available on system, it switches to CPU.
    num_workers : int, optional
        Number of workers for DataLoader, by default DEFAULTS.num_workers
    data_parallel : Union[list, None], optional
        If not None, this specifies the device ids for DataParallel mode in Pytorch.
        See https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html.
    log_every_k_steps: Union[int, None], optional
        If not None, log batch loss and grad_norm every k steps. Careful: this results in larger output files.
        By default None (no stepwise logging).
    verbose : bool, optional
        Verbose mode flag.
        If True, prints progress bars, model architecture and other useful information.
    force_deterministic : bool, optional
        Whether to run in Pytorch (full) deterministic mode.
        Not recommended, as this leads to substantial slow down. Seeds are set also without setting this to True.
    """
    
    # load config
    Conf = ConfigManager(exp_id=exp_id, config_dir=config_dir)
    exp_list = Conf.create_config_list()
        
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
                 log_every_k_steps=log_every_k_steps,
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
            log_every_k_steps=args.log_every_k_steps,
            verbose=args.verbose,
            force_deterministic=args.force_deterministic)
    

