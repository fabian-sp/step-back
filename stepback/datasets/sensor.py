""" Loading sensor data for matrix completion.

See for reference:
    L.M. Rivera-Muñoz, A.F. Giraldo-Forero, J.D. Martinez-Vargas,
    Deep matrix factorization models for estimation of missing data in a low-cost sensor network to measure air quality,
    Ecological Informatics, Volume 71, 2022.


Seed for split into train, validation is fixed.

The dataset is taken from:
    https://github.com/andresgiraldo3312/DMF/blob/main/DatosEliminados/Ventana_Eli_mes1.csv

The dataloading procedure is adapted from:
    https://github.com/andresgiraldo3312/DMF/blob/main/DMF_1.ipynb   
"""
import torch
import numpy as np
import pandas as pd

SPLIT_SEED = 12345678

def get_sensor(split, path):
    db = pd.read_csv(path + 'sensor/Ventana_Eli_mes1.csv', sep=',', index_col=0)
    X = np.asarray(db) 

    rows, cols = np.nonzero(X)          # Use only nonzero entries                                                               
    indices = np.zeros((len(rows),2))   # sensor, time (row, column)
    values = np.zeros(len(rows))
    
    indices[:,0] = rows
    indices[:,1] = cols
    values = X[rows,cols]
    
    # permute dataset with fixed seed
    rng0 = np.random.RandomState(SPLIT_SEED)
    indices = rng0.permutation(indices)
    rng0 = np.random.RandomState(SPLIT_SEED)
    values = rng0.permutation(values)
                                                                    
    N_train = int(0.8*len(rows))                                                                   
    n_sensors, n_time = X.shape
    
    # de-mean and scale (only with train set info)
    mean_ = values[0:N_train].mean()
    std_ = values[0:N_train].std()
    
    values =  (values-mean_)/std_ # standardize
    
    torch.testing.assert_allclose(torch.tensor(values[0:5]), torch.tensor([-0.6056, -0.4624, 0.2748, -0.5206, -0.6135]), rtol=1e-3, atol=1e-3)
    
    if split == 'train':
        ds = torch.utils.data.TensorDataset(torch.IntTensor(indices[0:N_train,:]), torch.Tensor(values[0:N_train, None])) 
    else:
        ds = torch.utils.data.TensorDataset(torch.IntTensor(indices[N_train:,:]), torch.Tensor(values[N_train:, None])) 
    
    # store
    ds.extra_info = dict() 
    ds.extra_info["dim"] = (n_sensors, n_time)
    ds.extra_info["scale"] = (mean_, std_)
    
    if split == 'train':
        ds.extra_info["true_matrix"] = (X-mean_)/std_

    return ds