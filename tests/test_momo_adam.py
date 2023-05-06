from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal
import torch
import numpy as np
import copy

from stepback.base import Base

name = 'test'
device = 'cpu'


_config = {"dataset": 'synthetic_linear',
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
    config = copy.deepcopy(_config)
    B = Base(name, config, device)
    B.setup()
    B.run()
    
    assert_almost_equal(B.results['history'][0]['train_loss'], 0.2505902022123337, decimal=5)
    assert_almost_equal(B.results['history'][0]['val_score'], 0.22642181515693666, decimal=5)
    goal_step_sizes = np.array([1.3795, 0.655501, 0.425859, 0.143845, 0.155355])
    assert_array_almost_equal(B.results['history'][0]['step_size_list'], goal_step_sizes, decimal=5)

    return

def test_momo_adam_weight_decay():    
    torch.manual_seed(123)
    config = copy.deepcopy(_config)
    config['opt']['weight_decay'] = 0.01
    B = Base(name, config, device)
    B.setup()
    B.run()
    
    assert_almost_equal(B.results['history'][0]['train_loss'], 0.24365038871765138, decimal=5)
    assert_almost_equal(B.results['history'][0]['val_score'], 0.22479842305183412, decimal=5)
    goal_step_sizes = np.array([1.38299, 0.653872, 0.420978, 0.142662, 0.158556])
    assert_array_almost_equal(B.results['history'][0]['step_size_list'], goal_step_sizes, decimal=5)

    return