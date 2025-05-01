#!/usr/bin/env python 

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, rdFingerprintGenerator
from tqdm import tqdm
from time import perf_counter


def ecfp4_generator(L=2048, use_chirality=True):
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=L, includeChirality=use_chirality)

def make_dataset_ecfp4_mpi(datapath, L=2048, N=None, smiles_col='SMILES_str', pce_col='pce'):

    filetype = datapath.strip().split('.')[-1]
    if filetype == 'csv':
        df = pd.read_csv(datapath, usecols=[smiles_col, pce_col])
    elif filetype == 'xlsx':
        df = pd.read_excel(datapath, usecols=[smiles_col, pce_col])
    else:
        print(f'File type .{filetype} not supported.')
        return None, None

    if N is None:
        N = len(df)

    X = np.zeros((N,L),dtype=np.uint8)
    y = np.zeros(N)

    fp_gen = ecfp4_generator(L=L)

    start = perf_counter()
    for k, dfrow in tqdm(df.iterrows(), total=N):
        if k == N:
            break
        try:
            smiles = dfrow[smiles_col]
            molecule = AllChem.MolFromSmiles(smiles)
            if molecule is not None:
                X[k, :] = fp_gen.GetCountFingerprintAsNumPy(molecule)
                y[k] = dfrow[pce_col]
            else:
                y[k] = -np.inf
        except Exception as e:
            print(f"Error at row {k} -> {e}",flush=True)
        end = perf_counter()
    print(f'{N} rows took {end-start} seconds ---> {(end-start)/N} seconds/row',flush=True)

    invalid = (y == -np.inf) # mask which selects all rows that threw an exxeption
    print(f'Found {invalid.sum()} invalid rows.', flush=True)
    return X[~invalid,:], y[~invalid]



print('Yeboiii', flush=True)

csvpath = 'moldata-filtered.csv'

X, y = make_dataset_ecfp4_mpi(csvpath)
np.save(f'ecfp4.npy', X)
np.save(f'pces.npy', y)
