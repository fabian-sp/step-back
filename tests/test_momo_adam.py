from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal
import torch
import numpy as np
import copy

from stepback.base import Base

name = 'test'
device = 'cpu'


config = {"dataset": 'synthetic_linear',
          "dataset_kwargs": {'p': 10, 'n_samples': 100},
          "model": 'linear',
          "loss_func": 'squared',
          "score_func": 'squared',
          "opt": {'name': 'momo-adam', 'lr': 1, 'weight_decay': 0, 'lr_schedule': 'constant'},
          "batch_size": 20,
          "max_epoch": 10,
          "run_id": 0}

def test_momo_adam():
    torch.manual_seed(123)
    B = Base(name, config, device)
    B.setup()
    B.run()
    
    assert_almost_equal(B.results['history'][0]['train_loss'], 0.5928951382637024, decimal=5)
    assert_almost_equal(B.results['history'][0]['val_score'], 0.5384113848209381, decimal=5)
    goal_step_sizes = np.array([0.13795, 0.00745714, 0.00856603, 0.00412777, 0.0151196])
    assert_array_almost_equal(B.results['history'][0]['step_size_list'], goal_step_sizes, decimal=5)

    return

def test_momo_adam_weight_decay():    
    torch.manual_seed(123)
    config2 = copy.deepcopy(config)
    config2['opt']['weight_decay'] = 0.01
    B = Base(name, config2, device)
    B.setup()
    B.run()
    
    assert_almost_equal(B.results['history'][0]['train_loss'], 0.5717200577259064, decimal=5)
    assert_almost_equal(B.results['history'][0]['val_score'], 0.5319624423980713, decimal=5)
    goal_step_sizes = np.array([0.138299, 0.00783268, 0.0089153, 0.00460468, 0.0153383])
    assert_array_almost_equal(B.results['history'][0]['step_size_list'], goal_step_sizes, decimal=5)

    return