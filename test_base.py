"""
Test Base class.
"""
import torch

from stepback.base import Base


# config = {"dataset": 'synthetic_matrix_fac',
#           "dataset_kwargs": {'p': 10, 'q': 5, 'n_samples': 100},
#           "model": 'matrix_fac',
#           "loss_func": 'squared',
#           "score_func": 'squared',
#           "opt": {'name': 'momo', 'lr': 1., 'weight_decay': 0, 'lr_schedule': 'sqrt', 'bias_correction': False},
#           "batch_size": 20,
#           "max_epoch": 30,
#           "run_id": 0}

config = {"dataset": 'synthetic_linear',
          "dataset_kwargs": {'p': 10, 'n_samples': 100},
          "model": 'linear',
          "loss_func": 'logistic',
          "score_func": 'logistic_accuracy',
          #"opt": {'name': 'sgd', 'lr': 1e-1,  'lr_schedule': 'sqrt'},
          "opt": {'name': 'momo', 'lr': 1e-1,  'lr_schedule': 'sqrt', 'bias_correction': False},
          "batch_size": 20,
          "max_epoch": 10,
          "run_id": 0}

name = 'test'
device = 'cpu'

B = Base(name, config, device)

B.config

B.setup()
B.run()
B.results

B.save_checkpoint(path='output/checkp/')
