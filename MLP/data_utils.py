
import numpy as np
from numpy.random import default_rng
import torch

def split_dataset(X,y,train_frac=0.8,valid_frac=0.0, seed=64):
    """Split data set into training and a test set."""

    assert train_frac + valid_frac < 1.0, f"train_frac ({train_frac}) + valid_frac ({valid_frac}) must be strictly smaller than 1!"

    rng = default_rng(seed)

    N = X.shape[0]
    shuffled = rng.permutation(N)

    train_frac = 0.8
    Ntrain = int(N * train_frac)

    Xtrain = X[shuffled[:Ntrain]]
    ytrain = y[shuffled[:Ntrain]]
    
    if valid_frac == 0.0:
        Xtest = X[shuffled[Ntrain:]]
        ytest = y[shuffled[Ntrain:]]
    
        return Xtrain, ytrain, Xtest, ytest
    else:
        Nvalid = int(N * valid_frac)
        itest0 = Ntrain + Nvalid

        Xvalid = X[shuffled[Ntrain:itest0]]
        yvalid = y[shuffled[Ntrain:itest0]]

        Xtest = X[shuffled[itest0:]]
        ytest = y[shuffled[itest0:]]

        return Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest


def np2torch(arrs):
    out = []
    for arr in arrs:
        if arr.dtype == np.uint8:
            out.append(torch.tensor(arr,dtype=torch.uint8))
        else:
            out.append(torch.tensor(arr))
    return out

def prepare_data(ecfp4_npy, pce_npy, train_frac=0.8, valid_frac=0.0):
    X = np.load(ecfp4_npy, mmap_mode='r')
    y = np.load(pce_npy, mmap_mode='r')
    
    arrs = split_dataset(X,y,train_frac=train_frac, valid_frac=valid_frac)
    torch_arrs = np2torch(arrs)

    return (*torch_arrs,)