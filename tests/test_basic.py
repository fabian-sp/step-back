from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal

from src.base import Base

config = {"dataset": 'synthetic_linear',
          "dataset_kwargs": {'p': 10, 'n_samples': 100},
          "model": 'linear',
          "loss_func": 'logistic',
          "score_func": 'logistic_accuracy',
          "opt": {'name': 'sgd', 'lr': 1e-1, 'weight_decay': 0, 'lr_schedule': 'sqrt'},
          "batch_size": 20,
          "max_epoch": 10,
          "run_id": 0}

name = 'test'
device = 'cpu'

def test_base_object():    
    B = Base(name, config, device)
    B.setup()
    B.run()
    B.results
    
    assert_almost_equal(B.results['history'][-1]['train_loss'], 0.44169596433639524, decimal=5)
    assert_almost_equal(B.results['history'][-1]['val_score'], 0.7500000119209289, decimal=5)
    assert_almost_equal(B.results['history'][-1]['model_norm'], 0.7555383443832397, decimal=5)

    return