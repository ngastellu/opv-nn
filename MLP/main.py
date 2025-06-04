import json
from pathlib import Path
import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import FingerprintMLPRegressor
from data_utils import prepare_data
from loss_fn import Loss
from trainer import train_model


def get_config():
    parser = argparse.ArgumentParser(description="Train a PCE regression MLP")

    # Data and logging
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--ecfp4_npy', type=str, default='data/ecfp4.npy')
    parser.add_argument('--pce_npy', type=str, default='data/pces.npy')

    # Model hyperparameters
    parser.add_argument('--input_dim', type=int, default=2048)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[4096, 1024])
    parser.add_argument('--dropout', type=float, default=0.3)

    # Training settings
    parser.add_argument('--tr_batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=-1) # if -1, do not implement early stopping

    # Testing settings
    parser.add_argument('--te_batch_size', type=int, default=64)

    # Loss and metrics
    parser.add_argument('--loss_func', choices=['mse', 'l1', 'huber'], default='mse')
    parser.add_argument('--huber_delta', type=float, default=0.1)

    return parser.parse_args()


def save_args(configs):
    """Saves run arguments to JSON file to keep track of experimens/ensure reproducibility."""
    outdir = Path(configs.experiment_name)
    outdir.mkdir(exist_ok=True)
    json_file = outdir / 'train_configs.json'
    print(f'Saving args to {json_file}...',end = ' ', flush=True)
    args_dict = vars(configs)

    with open(json_file, 'w') as fo:
        json.dump(args_dict, fo, indent=4)
    print('Done!', flush=True)


def init_model(configs):
    model = FingerprintMLPRegressor(configs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device

def setup_training(configs, Xtr, ytr, Xte, yte, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.learning_rate, weight_decay=configs.weight_decay)
    loss_func = Loss(configs)

    # convert everything to torch data types
    Xtr = torch.from_numpy(Xtr).float()
    ytr = torch.from_numpy(ytr).float()
    Xte = torch.from_numpy(Xte).float()
    ytr = torch.from_numpy(ytr).float()

    train_dataset = TensorDataset(Xtr, ytr)
    train_loader = DataLoader(train_dataset, batch_size=configs.tr_batch_size, shuffle=True)
    
    test_dataset = TensorDataset(Xte, yte)
    test_loader = DataLoader(test_dataset, batch_size=configs.val_batch_size, shuffle=True)
    return optimizer, loss_func, train_loader, test_loader 

def main(configs):
    model, device  = init_model(configs)

    Xtr, ytr, Xte, yte = prepare_data(configs.ecfp4_npy, configs.pce_npy)
    optimizer, loss_func, train_loader, test_loader = setup_training(configs, Xtr, ytr, Xte, yte, model)
    train_model(configs, model, train_loader, test_loader, optimizer, loss_func, device)


if __name__ == "__main__":
    configs = get_config()
    save_args(configs)
    main(configs)
