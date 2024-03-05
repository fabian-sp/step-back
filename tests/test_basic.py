from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal

from stepback.base import Base

name = 'test'
device = 'cpu'


config = {"dataset": 'synthetic_linear',
          "dataset_kwargs": {'p': 10, 'n_samples': 100},
          "model": 'linear',
          "loss_func": 'logistic',
          "score_func": 'logistic_accuracy',
          "opt": {'name': 'sgd', 'lr': 1e-1, 'weight_decay': 0, 'lr_schedule': 'sqrt'},
          "batch_size": 20,
          "max_epoch": 10,
          "run_id": 0}

def test_base_object():    
    B = Base(name, config, device)
    B.setup()
    B.run()
    
    assert_almost_equal(B.results['history'][-1]['train_loss'], 0.44169596433639524, decimal=3)
    assert_almost_equal(B.results['history'][-1]['val_score'], 0.7500000119209289, decimal=3)
    assert_almost_equal(B.results['history'][-1]['model_norm'], 0.7555383443832397, decimal=3)

    return


config_logreg = {"dataset": 'synthetic_linear',
                  "dataset_kwargs": {'p': 5, 'n_samples': 100},
                  "model": 'linear',
                  "loss_func": 'logistic',
                  "score_func": 'logistic_accuracy',
                  "opt": {'name': 'sgd', 'lr': 1., 'weight_decay': 1e-2, 'lr_schedule': 'constant'},
                  "batch_size": 100,
                  "max_epoch": 200,
                  "run_id": 0}

def test_logreg():
    """Solve Logistic regression with Gradient Descent."""
    B = Base(name, config_logreg, device)
    B.setup()
    B.run()
    
    lam = B.config['opt']['weight_decay']
    f1 = B.results['history'][-1]['train_loss'] + (lam/2) * B.results['history'][-1]['model_norm']**2
    f2 = B.results['summary']['opt_val']
    
    assert_almost_equal(f1, f2, decimal=3)

    return

config_ridge = {"dataset": 'synthetic_linear',
                  "dataset_kwargs": {'p': 10, 'n_samples': 100},
                  "model": 'linear',
                  "loss_func": 'squared',
                  "score_func": 'squared',
                  "opt": {'name': 'sgd', 'lr': 1e-1, 'weight_decay': 1e-3, 'lr_schedule': 'constant'},
                  "batch_size": 100,
                  "max_epoch": 100,
                  "run_id": 0}

def test_ridge():
    """Solve Ridge regerssion with Gradient Descent."""
    B = Base(name, config_ridge, device)
    B.setup()
    B.run()
    
    lam = B.config['opt']['weight_decay']
    f1 = B.results['history'][-1]['train_loss'] + (lam/2) * B.results['history'][-1]['model_norm']**2
    f2 = B.results['summary']['opt_val']
        
    assert_almost_equal(f1, f2)

    return

config_ridge_bias = {"dataset": 'synthetic_linear',
                    "dataset_kwargs": {'p': 10, 'n_samples': 100},
                    "model": 'linear',
                    "model_kwargs": {"bias": True},
                    "loss_func": 'squared',
                    "score_func": 'squared',
                    "opt": {'name': 'sgd', 'lr': 1e-1, 'weight_decay': 1e-3, 'lr_schedule': 'constant'},
                    "batch_size": 100,
                    "max_epoch": 1000,
                    "run_id": 0}

def test_ridge_bias():
    """Solve Ridge regerssion with Gradient Descent."""
    B = Base(name,config_ridge_bias, device)
    B.setup()
    B.run()
    
    lam = B.config['opt']['weight_decay']
    f1 = B.results['history'][-1]['train_loss'] + (lam/2) * B.results['history'][-1]['model_norm']**2
    f2 = B.results['summary']['opt_val']
        
    assert_almost_equal(f1, f2)

    return

config_logreg_bias = {"dataset": 'synthetic_linear',
                    "dataset_kwargs": {'p': 5, 'n_samples': 100},
                    "model": 'linear',
                    "model_kwargs": {"bias": True},
                    "loss_func": 'logistic',
                    "score_func": 'logistic_accuracy',
                    "opt": {'name': 'sgd', 'lr': 1., 'weight_decay': 1e-2, 'lr_schedule': 'constant'},
                    "batch_size": 100,
                    "max_epoch": 300,
                    "run_id": 0}

def test_logreg_bias():
    """Solve Logistic regression with Gradient Descent."""
    B = Base(name, config_logreg_bias, device)
    B.setup()
    B.run()
    
    lam = B.config['opt']['weight_decay']
    f1 = B.results['history'][-1]['train_loss'] + (lam/2) * B.results['history'][-1]['model_norm']**2
    f2 = B.results['summary']['opt_val']
    
    assert_almost_equal(f1, f2, decimal=3)

    return