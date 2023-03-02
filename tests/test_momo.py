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
          "opt": {'name': 'momo', 'lr': 1, 'weight_decay': 0, 'lr_schedule': 'constant'},
          "batch_size": 20,
          "max_epoch": 10,
          "run_id": 0}

def test_momo_bias():
    torch.manual_seed(123)

    config = copy.deepcopy(_config)
    config['opt']['bias_correction'] = True
    B = Base(name, config, device)
    B.setup()
    B.run()
    
    assert_almost_equal(B.results['history'][0]['train_loss'], 0.26801398396492004, decimal=5)
    assert_almost_equal(B.results['history'][0]['val_score'], 0.22546180486679077, decimal=5)
    goal_step_sizes = np.array([1.00637, 0.60069, 0.46531, 0.20478, 0.2002])
    assert_array_almost_equal(B.results['history'][0]['step_size_list'], goal_step_sizes, decimal=5)

    return

def test_momo():  
    torch.manual_seed(123)

    config = copy.deepcopy(_config)
    config['opt']['bias_correction'] = False
    B = Base(name, config, device)
    B.setup()
    B.run()
    
    assert_almost_equal(B.results['history'][0]['train_loss'], 0.632905125617981, decimal=5)
    assert_almost_equal(B.results['history'][0]['val_score'], 0.5792657136917114, decimal=5)
    goal_step_sizes = np.array([0.10064, 0.0056, 0.00647, 0.00352, 0.01141])
    assert_array_almost_equal(B.results['history'][0]['step_size_list'], goal_step_sizes, decimal=5)

    return

def test_momo_weight_decay():  
    torch.manual_seed(123)

    config = copy.deepcopy(_config)
    config['opt']['weight_decay'] = 0.1
    B = Base(name, config, device)
    B.setup()
    B.run()
    
    assert_almost_equal(B.results['history'][0]['train_loss'], 0.49298152327537537, decimal=5)
    assert_almost_equal(B.results['history'][0]['val_score'], 0.5650612115859985, decimal=5)
    goal_step_sizes = np.array([0.103182, 0.00838149, 0.00942883, 0.00738135, 0.0144944])
    assert_array_almost_equal(B.results['history'][0]['step_size_list'], goal_step_sizes, decimal=5)

    return