"""
Debug Base class.
"""
import torch
import os

os.chdir('..')
from stepback.base import Base

# config = {"dataset": "sensor",
#           "model": "matrix_completion",
#           "model_kwargs": {"dim": [130,720], "rank": 24},
#           "loss_func": "squared",
#           "score_func": "squared",
#           "opt": {'name': 'sgd', 'lr': 1, 'weight_decay': 1e-4, 'lr_schedule': 'constant'},
#           #"opt": {'name': 'momo', 'lr': 1, 'weight_decay': 1e-4, 'lr_schedule': 'constant'},
#           "batch_size": 128,
#           "max_epoch": 50,
#           "run_id": 0}

config = {"dataset": "cifar10",
          "model": "vit",
          "loss_func": "cross_entropy",
          "score_func": "cross_entropy_accuracy",
          "opt": {'name': 'adamw', 'lr': 1e-3, 'weight_decay': 0.1, 'lr_schedule': 'sqrt'},
          #"opt": {'name': 'momo', 'lr': 1, 'weight_decay': 1e-4, 'lr_schedule': 'constant'},
          "batch_size": 128,
          "max_epoch": 1,
          "run_id": 0}

name = 'test'
device = 'cpu'

B = Base(name, config, device)

B.config

B.setup()
B.run()
B.results

# B.save_checkpoint(path='output/checkp/')
