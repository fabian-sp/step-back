from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal
import torch
import numpy as np

from stepback.base import Base

name = 'test'
device = 'cpu'

torch.manual_seed(123)

config = {"dataset": 'synthetic_linear',
          "dataset_kwargs": {'p': 10, 'n_samples': 100},
          "model": 'linear',
          "loss_func": 'squared',
          "score_func": 'squared',
          "opt": {'name': 'momo', 'lr': 1, 'weight_decay': 0, 'lr_schedule': 'constant'},
          "batch_size": 20,
          "max_epoch": 10,
          "run_id": 0}

def test_momo():    
    B = Base(name, config, device)
    B.setup()
    B.run()
    
    assert_almost_equal(B.results['history'][0]['train_loss'], 0.26801398396492004, decimal=5)
    assert_almost_equal(B.results['history'][0]['val_score'], 0.22546180486679077, decimal=5)
    goal_step_sizes = np.array([1.00637, 0.60069, 0.46531, 0.20478, 0.2002])
    assert_array_almost_equal(B.results['history'][0]['step_size_list'], goal_step_sizes, decimal=5)

    return

def test_momo_no_bias():    
    config2 = config.copy()
    config2['opt']['bias_correction'] = False
    B = Base(name, config, device)
    B.setup()
    B.run()
    
    assert_almost_equal(B.results['history'][0]['train_loss'], 0.632905125617981, decimal=5)
    assert_almost_equal(B.results['history'][0]['val_score'], 0.5792657136917114, decimal=5)
    goal_step_sizes = np.array([0.10064, 0.0056, 0.00647, 0.00352, 0.01141])
    assert_array_almost_equal(B.results['history'][0]['step_size_list'], goal_step_sizes, decimal=5)

    return

def test_momo_weight_decay():    
    config2 = config.copy()
    config2['opt']['weight_decay'] = 0.1
    B = Base(name, config, device)
    B.setup()
    B.run()
    
    assert_almost_equal(B.results['history'][0]['train_loss'], 0.49726436734199525, decimal=5)
    assert_almost_equal(B.results['history'][0]['val_score'], 0.5778649687767029, decimal=5)
    goal_step_sizes = np.array([0.10064, 0.00784, 0.00874, 0.00667, 0.01315])
    assert_array_almost_equal(B.results['history'][0]['step_size_list'], goal_step_sizes, decimal=5)

    return