from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

import os
import torch
import numpy as np
import urllib

SPLIT_SEED = 12345678

LIBSVM_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"

# mapping libsvm names to download links
LIBSVM_NAME_MAP = {"rcv1"           : "rcv1_train.binary.bz2",
                    "mushrooms"     : "mushrooms",
                    "a1a"           : "a1a",
                    "ijcnn"         : "ijcnn1.tr.bz2", 
                    "breast-cancer" : "breast-cancer_scale"
                    }


def get_libsvm(split, name, path, train_size=0.8):
    X, y = load_libsvm(name, path + '/libsvm')
        
    if np.all(np.isin(y, [0,1])):
        y = y*2 - 1 # go from 0,1 to -1,1
    
    if name == 'breast-cancer':
        y[y==2] = 1
        y[y==4] = -1
    
    labels = np.unique(y)
    assert np.all(np.isin(y, [-1,1])), f"Sth went wrong with class labels, have {labels}."
    
    # use fixed seed for train/val split
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, 
                                                        train_size=train_size, 
                                                        shuffle=True, 
                                                        random_state=SPLIT_SEED)
    
    if split == 'train':
        X = torch.FloatTensor(X_train.toarray())
        Y = torch.FloatTensor(Y_train)
    else:
        X = torch.FloatTensor(X_test.toarray())
        Y = torch.FloatTensor(Y_test)
    
    ds = torch.utils.data.TensorDataset(X, Y)
        
    return ds



def load_libsvm(name, path):
    if not os.path.exists(path):
        os.mkdir(path)

    fn = LIBSVM_NAME_MAP[name]
    filename = os.path.join(path, fn)

    if not os.path.exists(filename):
        url = urllib.parse.urljoin(LIBSVM_URL, fn)
        print("Downloading from %s" % url)
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")

    X, y = load_svmlight_file(filename)
    return X, y

