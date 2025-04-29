
from ase.db import connect
import pandas as pd
import numpy as np
from numpy.random import default_rng
from rdkit.Chem import AllChem, rdFingerprintGenerator
import matplotlib.pyplot as plt




def get_row_data(df,mol_type,mol_name):
    if mol_name == 'PC61BM':
        return {'LUMO': -3.70, 'LUMO+1': -3.70 + 0.077824564}
    elif mol_name == 'PC71BM':
        return {'LUMO': -3.91, 'LUMO+1': 0.033470005}

    filtered = df[df[mol_type] == mol_name]

    if filtered.empty:
        print(f'{mol_type} {mol_name} is missing from Excel sheet')
        return None
    else:
        return filtered.iloc[0].to_dict()


def get_don_acc_combined_features(don_homo,don_lumo,acc_lumo):
    edahl = acc_lumo - don_homo
    edall = don_lumo - acc_lumo
    return edahl, edall


def make_indoor_data(device_xlsx, donor_csv, acceptor_csv, normalize=False, minmax_scale=True):
    don_df = pd.read_csv(donor_csv)
    acc_df = pd.read_csv(acceptor_csv)
    df = pd.read_excel(device_xlsx)

    ndevices = len(df)
    X = np.zeros((ndevices,8))
    y = np.ones(ndevices) * -1.0

    seen_devices = set() #set of pairs of acceptor-donor pairs; avoids conflicting data points

    for k, row in df.iterrows():
        donor = row['Donor'].strip()
        acceptor = row['Acceptor'].strip()
        device = (donor,acceptor)
        
        if device in seen_devices:
            print('Already saw ', device)
            continue
        
        seen_devices.add(device)

        don_data = get_row_data(don_df,'Donors', donor)
        acc_data = get_row_data(acc_df,'Acceptors', acceptor)

        if don_data is None or acc_data is None:
            continue

        don_homo = don_data['HOMO']
        don_lumo = don_data['LUMO']
        don_dhomo = don_homo - don_data['HOMO-1']
        don_dlumo = don_data['LUMO+1'] - don_lumo
        don_et1 = don_data['DET1']
        don_nd = don_data['Nd']
        
        acc_lumo = acc_data['LUMO']
        # adl = acc_data['LUMO+1'] - acc_lumo

        edahl, edall = get_don_acc_combined_features(don_homo,don_lumo,acc_lumo)

        X[k,0] = don_homo
        X[k,1] = don_lumo
        X[k,2] = don_et1
        X[k,3] = don_dhomo
        X[k,4] = don_dlumo
        X[k,5] = float(don_nd)
        X[k,6] = edahl
        X[k,7] = edall
        # X[k,8] = adl

        y[k] = row['PCE(%)']

    good_row_filter = y>=0 # True for rows with no missing data
    X = X[good_row_filter,:]
    y = y[good_row_filter]
    
    # Rescales data to be between 0 and 1
    # Rescale before standardizing?
    if minmax_scale:
         X = (X - np.min(X,axis=0)) / (np.max(X,axis=0) - np.min(X, axis=0))

    # Standardize (zero-mean, unit varianc)
    if normalize:
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
    

    return X,y


def cal_prop_cc(molo,tag):
    if molo.data.Acceptor=='PC61BM':
        al= -3.70
        adl= 0.077824564
    elif molo.data.Acceptor=='PC71BM':
        al= -3.91
        adl= 0.033470005
    if tag=='edahl':
        prop=al-float(molo.homo)
    if tag=='edall':
        prop=float(molo.lumo)-al
    if tag=='adlumo':
        prop=adl
    if tag=='nd':
        # prop=cal_nd(moln)
        prop = float(molo.nd)
    return prop


def make_cc_data(path_to_db, normalize=False, minmax_scale=True):
    db = connect(path_to_db)
    nmols = db.count()
    X = np.zeros((nmols,8))
    y = np.zeros(nmols)
    
    for k, row in enumerate(db.select()):
        print(row.data)
        X[k,0] = row.homo
        X[k,1] = row.lumo
        X[k,2] = row.et1
        X[k,3] = row.dh
        X[k,4] = row.dl
        X[k,5] = row.nd
        X[k,6] = cal_prop_cc(row,'edahl') 
        X[k,7] = cal_prop_cc(row,'edall') 

        y[k] = row.data.PCE
    

    # Rescales data to be between 0 and 1
    # Rescale before standardizing?
    if minmax_scale:
         X = (X - np.min(X,axis=0)) / (np.max(X,axis=0) - np.min(X, axis=0))

    # Standardize (zero-mean, unit varianc)
    if normalize:
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
 
    return X, y


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

def standardize(X):
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
        return X

def minmax_scale(X):
    # Rescales data to be between 0 and 1
    # Rescale before standardizing?
        X = (X - np.min(X,axis=0)) / (np.max(X,axis=0) - np.min(X, axis=0))
        return X
    
def split_data_db(db, standardize=False, minmax_scale=False,train_frac=0.8,valid_frac=0.0):
    # Define inputs to model
    feature_keys = ['homo', 'lumo', 'et1', 'dhomo', 'dlumo', 'nd', 'edahl', 'edall', 'adlumo', 'pce']
    data = np.array([[row.id] + [row.data[fk] for fk in feature_keys] for row in db.select()])

    X = data[:,1:-1]
    y = data[:,-1]

    if standardize:
        X = standardize(X)
    
    if minmax_scale:
        X = minmax_scale(X)

    Xtrain, ytrain, Xtest, ytest = split_dataset(X,y,train_frac=train_frac, valid_frac=valid_frac)

    return Xtrain, ytrain, Xtest, ytest, feature_keys

# constructs an ECFP4 fingerprint generator
def ecfp4_generator(L=2048,use_chirality=True):
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=L, includeChirality=use_chirality)


def make_dataset_ecfp4(datapath, L=2048, N=None,smiles_col='SMILES_str',pce_col='pce'):
    filetype = datapath.strip().split('.')[-1]
    print(filetype)
    if filetype == 'csv':
        df = pd.read_csv(datapath, usecols=[smiles_col, pce_col])
    elif filetype == 'xlsx':
        df = pd.read_excel(datapath, usecols=[smiles_col, pce_col])
    else:
        print(f'File type .{filetype} not  supported. Returning 0 in confusion.')
        return 0

    if N is None:
        N = len(df)

    X = np.zeros((N,L), dtype=np.uint8) # molecular fingerprints
    y = np.zeros(N) # PCEs

    fp_gen = ecfp4_generator(L=L)

    for k, row in df.iterrows(): 
        if k == N:
            break
        # print(f'\n******* {k} *******')
        try:
            molecule = AllChem.MolFromSmiles(row[smiles_col])
            X[k,:] = fp_gen.GetCountFingerprintAsNumPy(molecule)
        except Exception as e:
            print(e)
            print(f'Skipping this row (k = {k})')
        y[k] = row[pce_col]
    
    # if pce_col == 'PCE(%)':
    #     y /= 100.0
    
    return X, y

    
def plot_predictions(y, y_pred, color=None, markersize=5.0,target_name='',plt_objs=None,show=True):
    ymin = np.min(y)
    ymax = np.max(y)
    reference = np.linspace(ymin, ymax, 100)

    if plt_objs is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt_objs
    
    ax.plot(reference, reference, 'k--', lw=0.8) # y = x to guide the eye
    ax.plot(y,y_pred, 'o', c=color, ms=markersize)
    ax.set_xlabel(f'True {target_name}')
    ax.set_ylabel(f'Predicted {target_name}')

    if show:
        plt.show()


