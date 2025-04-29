import argparse
import csv
import os
import sys
from random import sample
from data.build_graph import *
from models.model import Net
from utils.logger import logger
from utils.utils import train_val_test_split

parser = argparse.ArgumentParser(description='SLI-GNN')
# parser.add_argument('trained_model', help='name of the trained model (weights/trained_model).')
# parser.add_argument('dataset_name', help='dataset name (data/dataset/dataset_name).')
# parser.add_argument('filename', metavar='F', help='csv filename(dataset/targets/filename.csv)')
parser.add_argument('-b', '--batch-size', default=1000, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=0.6, type=float, metavar='N',
                        help='number of training data to be loaded (default 0.6)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                        help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--valid-ratio', default=0.2, type=float, metavar='N',
                        help='percentage of validation data to be loaded (default '
                            '0.2)')
valid_group.add_argument('--valid-size', default=None, type=int, metavar='N',
                        help='number of validation data to be loaded (default '
                            '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.2, type=float, metavar='N',
                        help='percentage of test data to be loaded (default 0.2)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')


# args = parser.parse_args(sys.argv[1:])
args = parser.parse_args()

molprop = 'e_homo_alpha'
model_path = f'weights/{molprop}_checkpoint.pth.tar'
if os.path.isfile(model_path):
    logger.info("=> loading model params '{}'".format(model_path))
    model_checkpoint = torch.load(model_path,
                                  map_location=lambda storage, loc: storage)
    model_args = argparse.Namespace(**model_checkpoint['args'])
    logger.info("=> loaded model params '{}'".format(model_path))
else:
    logger.info("=> no model params found at '{}'".format(model_path))

best_loss = 1e10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def main():
    global args, model_args, best_loss, model_path, molprop

    # load data
    datapath = "D:/harvard-cep-dataset-main/Raw-data"
    path = os.path.join(datapath,'xyzfiles')
    is_db = path.split('.')[-1] == 'db' # check if reading input data from database

    # path = "/Users/nico/Desktop/scripts/OPVGCN/data/train.db"
    # path = 'data/dataset/'
    
   
    print(f'\n\n~~~~~~~~ Reading from {path} ~~~~~~~~')
    targets_filename = os.path.join(os.path.join(datapath,f'moldata_xyzexists_{molprop}.csv'))
    # targets_filename = f"data/dataset/targets/train_{mp}-targets.csv"
    print(f'** Target CSV =  {targets_filename} **')
    properties_list = model_args.properties[0]
    is_db = path.split('.')[-1] == 'db'
    # print(f'************** is_db = {is_db} **************')
    dataset = GraphData(path=path, targets_filename=targets_filename, max_num_nbr=model_args.max_num_nbr, radius=model_args.radius,
                        properties_list=properties_list, step=model_args.step,is_db=is_db)
    
    my_atom_ref = None


    *_, test_loader = train_val_test_split(dataset,
                             batch_size=args.batch_size,
                             train_ratio=args.train_ratio,
                             valid_ratio=args.valid_ratio,
                             test_ratio=args.test_ratio,
                             num_workers=args.workers,
                             train_size=args.train_size,
                             valid_size=args.valid_size,
                             test_size=args.test_size)
    
    Ndata = len(dataset)
    Ntest = len(test_loader)
    print(f'******* Size of complete dataset = {Ndata} *******')
    print(f'******* Size of test set = {Ntest} (should be = {Ndata*args.test_ratio}) *******')

    # build model
    orig_bond_fea_len = dataset.bond_feature_encoder.num_category

    model = Net(orig_bond_fea_len=orig_bond_fea_len,
                atom_fea_len=model_args.atom_fea_len,
                nbr_fea_len=model_args.nbr_fea_len,
                n_conv=model_args.n_conv,
                h_fea_len=model_args.h_fea_len,
                l1=model_args.l1, l2=model_args.l2,
                classification=True if model_args.task == 'classification' else False,
                n_classes=model_args.n_classes,
                attention=model_args.attention,
                dynamic_attention=model_args.dynamic_attention,
                n_heads=model_args.n_heads,
                max_num_nbr=model_args.max_num_nbr,
                pooling=model_args.pooling,
                p=model_args.dropout_p,
                properties_list=properties_list)
    model.to(device)

    # define loss func and optimizer
    if model_args.task == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    if model_args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        if len(dataset) < 500:
            sample_target = [dataset[i].y for i in range(len(dataset))]
        else:
            sample_target = [dataset[i].y for i in sample(range(len(dataset)), 500)]

        normalizer = Normalizer(torch.tensor(sample_target),atom_ref=my_atom_ref)

    # optionally resume from a checkpoint
    if os.path.isfile(model_path):
        logger.info("=> loading model '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        normalizer.load_state_dict(checkpoint['normalizer'])
        logger.info("=> loaded model '{}' (epoch {}, validation {})"
            .format(model_path, checkpoint['epoch'],
                    checkpoint['best_loss']))
    else:
        logger.info("=> no model found at '{}'".format(model_path))
    test(test_loader, model, criterion, normalizer,molprop)


def test(test_loader, model, criterion, normalizer,molprop):
    test_material_ids = []
    test_targets = []
    test_preds = []

    if model_args.task == 'classification':
        probabilities = []

    running_loss = AverageMeter()
    if model_args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
    for batch_idx, data in enumerate(test_loader, 0):
        # if batch_idx > 50: # early stopping
        #     break
        with torch.no_grad():
            if model_args.task == 'regression':
                targets = data.y.unsqueeze(1)
                targets_normed = normalizer.norm(targets)
            else:
                targets = data.y.long()
                targets_normed = targets
            data, targets_normed = data.to(device), targets_normed.to(device)

            outputs = model(data)

            loss = criterion(outputs, targets_normed)
            running_loss.update(loss.item(), targets.size(0))

            material_id = data.material_id
            test_target = targets

            test_material_ids += material_id

            if model_args.task == 'regression':
                test_pred = normalizer.denorm(outputs.data.cpu())
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
            else:
                probability = nn.functional.softmax(outputs, dim=1)
                probability = probability.tolist()

                prediction = outputs.cpu().detach().numpy()
                test_pred = np.argmax(prediction, axis=1)
                test_preds += test_pred.tolist()

                test_targets += test_target.view(-1).tolist()
                probabilities += probability

            if model_args.task == 'regression':
                mae = mae_metric(normalizer.denorm(outputs.data.cpu()), targets)
                mae_errors.update(mae, targets.size(0))
                if batch_idx % model_args.print_space == 0:
                    logger.info('batch_idx: %2d, loss: %.3f, MAE: %.3f' % (
                        batch_idx + 1, running_loss.avg, mae_errors.avg))
            else:
                accuracy = class_metric(outputs, targets)
                accuracies.update(accuracy, targets.size(0))
                if batch_idx % model_args.print_space == 0:
                    logger.info('batch_idx: %2d, loss: %.3f, accuracy: %.3f' % (
                        batch_idx + 1, running_loss.avg, accuracies.avg))

    if model_args.task == 'regression':
        with open(f'results/regression/test_{molprop}_results.csv', 'w') as f:
            writer = csv.writer(f)
            for material_id, pred, target in zip(test_material_ids, test_preds, test_targets):
                writer.writerow((material_id, round(pred, 2), target))

        df = pd.read_csv('results/regression/test_results.csv',
                        header=None,
                        names=['material_id', 'Prediction', 'Target'])
        df.to_csv('results/regression/test_results.csv', index=False)
    else:
        with open('results/classification/test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for material_id, pred, target, probability in zip(test_material_ids, test_preds, test_targets, probabilities):
                writer.writerow((material_id, round(pred, 2), target, probability))

        df = pd.read_csv('results/classification/test_results.csv',
                        header=None,
                        names=['material_id', 'Prediction', 'Target', 'Probabilities'])
        df.to_csv('results/classification/test_results.csv', index=False)

    return running_loss.avg


if __name__ == '__main__':
    main()
