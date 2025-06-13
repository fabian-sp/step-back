from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal
import torch
import numpy as np

from stepback.base import Base

config = {"dataset": 'cifar10',
          "model": 'resnet20',
          "loss_func": "cross_entropy",
          "score_func": "cross_entropy_accuracy",
          "batch_size": 20,
          "max_epoch": 10}

opt_configs = [
                {"name": "adam", "lr": 1e-2}, 
                {"name": "sgd", "lr": 1e-2},
                {"name": "momo", "lr": 1e-2}
                ]

run_ids = [0,1]

def _template_resnet_init(config):
    """test that model initialization is fixed"""
    
    B = Base('test_resnet', config, device='cpu')
    B._setup_model() # only load model as we do not want to download dataset

    B.model.conv1.weight
    expected = np.array([-0.0005,  0.1434,  0.2429, -0.4450,  0.1392])
    assert_array_almost_equal(B.model.conv1.weight[:5,0,0,0].detach().numpy(), expected, decimal=3)
    
    return

def test_resnet_init():
    for opt in opt_configs:
        for run_id in run_ids:
            this_config = config.copy()
            this_config['run_id'] = run_id
            this_config['opt'] = opt
        
            _template_resnet_init(this_config)
    return