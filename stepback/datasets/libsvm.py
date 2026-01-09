from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

import os
import torch
import numpy as np
import urllib

SPLIT_SEED = 12345678

LIBSVM_BINARY_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
LIBSVM_MULTICLASS_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/"

# mapping libsvm names to download links
MULTICLASS_NAME_MAP = {
    "dna"           : "dna.scale",
    "wine"          : "wine.scale",
    "vowel"         : "vowel.scale",
}

BINARY_NAME_MAP = {
    "rcv1"          : "rcv1_train.binary.bz2",
    "mushrooms"     : "mushrooms",
    "a1a"           : "a1a",
    "ijcnn"         : "ijcnn1.tr.bz2", 
    "breast-cancer" : "breast-cancer_scale",
}
LIBSVM_NAMES = list(MULTICLASS_NAME_MAP.keys()) + list(BINARY_NAME_MAP.keys())


def get_libsvm(split, name, path, train_size=0.8):
    if name in BINARY_NAME_MAP.keys():
        multiclass = False
    elif name in MULTICLASS_NAME_MAP.keys():
        multiclass = True
    else:
        raise KeyError(f"Unknwon dataset name {name} from LIBSVM. Need to be added to name mapping.")
    
    libsvm_path = os.path.join(path, "libsvm")
    X, y = load_libsvm(name, libsvm_path, multiclass)

    if not multiclass:
        # use -1, 1 labels in binary case
        if np.all(np.isin(y, [0,1])):
            y = y*2 - 1
        
        # manual label fix for breast cancer dataset
        if name == 'breast-cancer':
            y[y==2] = 1
            y[y==4] = -1
        
        unique_labels = [-1, 1]
        assert np.all(np.isin(y, unique_labels)), f"Sth went wrong with class labels, have {np.unique(y)}."
    else:
        unique_labels = list(np.unique(y).astype("int"))
        # try to achieve class labels [0,C)
        if min(unique_labels) > 0:
            y = y - min(unique_labels)

    print(f"Dataset labels (before split): {unique_labels}")

    # use fixed seed for train/val split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        y, 
        train_size=train_size, 
        shuffle=True, 
        random_state=SPLIT_SEED
    )

    # for multiclass, we need Long Tensors
    # for binary, FloatTensor is ok
    if split == 'train':
        X = torch.FloatTensor(X_train.toarray())
        Y = torch.FloatTensor(Y_train) if not multiclass else torch.LongTensor(Y_train)
    else:
        X = torch.FloatTensor(X_test.toarray())
        Y = torch.FloatTensor(Y_test) if not multiclass else torch.LongTensor(Y_test)
    
    ds = torch.utils.data.TensorDataset(X, Y)
        
    return ds


def load_libsvm(name, path, multiclass=False):
    if not os.path.exists(path):
        os.mkdir(path)

    if multiclass:
        fn = MULTICLASS_NAME_MAP[name]
    else:
        fn = BINARY_NAME_MAP[name]
    filename = os.path.join(path, fn)

    _url = LIBSVM_MULTICLASS_URL if multiclass else LIBSVM_BINARY_URL
    if not os.path.exists(filename):
        url = urllib.parse.urljoin(_url, fn)
        print("Downloading from %s" % url)
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")

    X, y = load_svmlight_file(filename)
    return X, y

