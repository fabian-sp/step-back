import numpy as np
import torch

_BASE_SEED = 12345678


def get_synthetic_matrix_fac(p: int, q: int, n_samples: int, noise: float=0, condition_number: float=1e-5, split: str='train', seed: int=1234):
    """
    Generate a synthetic matrix factorization dataset:
    Adapted from: https://github.com/benjamin-recht/shallow-linear-net/blob/master/TwoLayerLinearNets.ipynb.
    
    NOTE: we use logspace instead of linspace for the condition number.            
    """
    # Atrue always uses same seed
    # measurements X depend on seed (use different seed for each run in validation set)
    rng0 = np.random.RandomState(seed)
    rng1 = np.random.RandomState(_BASE_SEED)
            
    # this is the same as multiplying D@B where B is random and D is diagonal with values as below:
    Atrue = np.logspace(0, np.log10(condition_number), q).reshape(-1, 1) * rng1.rand(q, p)
    
    # perturb train set
    if split == 'train' and noise>0:
        E = noise * (rng1.rand(q, p)*2-1) # E is in noise*[-1,1]
        Atrue *= (1+E)

    # create data and targets
    X = rng0.randn(p, n_samples)
    Y = Atrue.dot(X) 
    
    ds = torch.utils.data.TensorDataset(torch.FloatTensor(X.T), torch.FloatTensor(Y.T))

    return ds

def get_synthetic_linear(p: int, n_samples: int, noise: float=0, condition_number=1e-5, classify: bool=True, split='train', seed=1234):
    """
    Generate a synthetic dataset for linear/logistic regression.
    """
    bias = 1
    scaling = 10 
    sparsity = 30 
    solutionSparsity = 1. # w is dense if equal 1
              
    # rng0 for different seeds, rng1 for fixed seed
    rng0 = np.random.RandomState(seed)
    rng1 = np.random.RandomState(_BASE_SEED)
    
    # oracle has fixed seed
    w = rng1.randn(p) * (rng1.rand(p) < solutionSparsity)
    assert np.abs(w).max() > 0 , "The synthetic data generator returns an oracle which is zero."
    
    # scale w to norm 1
    w = w/np.linalg.norm(w)
    
    # A should be different for train and test
    A = rng0.randn(n_samples, p) + bias
    A = A.dot(np.diag(scaling*rng0.randn(p)))
    A = A * (rng0.rand(n_samples, p) < (sparsity*np.log(n_samples)/n_samples))
    
    assert np.linalg.norm(A, axis=1).min() > 0 , "A has zero rows"
    
    column_norm = 10
    A = column_norm * A / np.linalg.norm(A, axis=0)[None,:].clip(min=1e-6) # scale A column-wise --> L_i are differrent
    
    # classification
    if classify:
        b = 2*(A.dot(w) >= 0).astype(int) - 1.
        b = b * np.sign(rng0.rand(n_samples)-noise)
        labels = np.unique(b)
        
        assert np.all(np.isin(b, [-1,1])), f"Sth went wrong with class labels, have {labels}"
        print(f"Labels (-1/1) with count {(b==-1).sum()}/{(b==1).sum()}.")
        
    # regression
    else:
        b = A.dot(w) + noise*rng0.randn(n_samples)            
        
    ds = torch.utils.data.TensorDataset(torch.FloatTensor(A), torch.FloatTensor(b))

    return ds
