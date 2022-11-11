"""
"""
import json
import copy
from itertools import product

from src.utils import prepare_config, create_exp_list

exp_id = 'test1'

with open(f'configs/{exp_id}.json') as f:
    exp_config = json.load(f)



exp_config = prepare_config(exp_config)
exp_list = create_exp_list(exp_config)
    

print(f"Created {len(exp_list)} different configurations.")


