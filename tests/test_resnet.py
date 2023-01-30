from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal
import torch
import numpy as np

from stepback.base import Base

config = {"dataset": 'cifar10',
          "model": 'resnet20',
          "loss_func": "cross_entropy",
          "score_func": "cross_entropy_accuracy",
          "opt": {"name": "adam", "lr": 1e-2},
          "batch_size": 20,
          "max_epoch": 10,
          "run_id": 0}

opt_configs = [
                {"name": "adam", "lr": 1e-2}, 
                {"name": "adam", "lr": 1e-1}, 
                {"name": "sgd", "lr": 1e-2}
                ]

def _template_resnet(opt):
    """test that model initialization is fixed"""
    config['opt'] = opt    
    B = Base('test_resnet', config)
    B.setup()

    B.model.conv1.weight
    goal = np.array([-0.0005,  0.1434,  0.2429, -0.4450,  0.1392])
    assert_array_almost_equal(B.model.conv1.weight[:5,0,0,0].detach().numpy(), goal, decimal=3)
    
    return

def test_resnet_init():
    for opt in opt_configs:
        _template_resnet(opt)
    return