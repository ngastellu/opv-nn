from pathlib import Path
from time import perf_counter
import torch
from early_stopping import EarlyStopping

def model_epoch(model, data_loader, optimizer, loss_func, device, update_gradients=True):
        start = perf_counter()
        if update_gradients:
            model.train()
        else:
            model.eval()

        avg_loss = 0.0
        nbatches = 0

        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            output = model(batch_x)
            loss = loss_func(output, batch_y) 

            avg_loss += loss.item()
            nbatches += 1

            if update_gradients:
                optimizer.zero_grad() # avoid accumulating gradients from previous epochs
                loss.backward() # compute gradients
                optimizer.step() # update params
        
        end = perf_counter()
        epoch_time = end - start
        avg_loss /= nbatches

        return avg_loss, epoch_time


def train_model(configs, model, train_loader, test_loader, optimizer, loss_func, device, iepoch=0, logmode='w'):
    stopper = EarlyStopping(configs.patience)
    rundir = Path(configs.experiment_name)
    logfile = rundir / 'training_log.txt'
    fo = open(logfile, logmode)

    if iepoch == 0:
        fo.write('epoch\ttraining loss\ttraining time [s]\ttest loss\ttest time [s]\n')
    
    stopped_early = False
    for epoch in range(iepoch, configs.max_epochs):
        tr_loss, tr_time = model_epoch(model, train_loader, optimizer, loss_func, device, update_gradients=True)
        te_loss, te_time = model_epoch(model, test_loader, optimizer, loss_func, device, update_gradients=False)

        fo.write(f'{epoch}\t{tr_loss:.4f}\t{tr_time:.2f}\t{te_loss:.4f}\t{te_time:.2f}\n')

        if epoch % configs.save_freq == 0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'tr_loss': tr_loss,
            'te_loss': te_loss
        }, Path / f'checkpoint-{epoch}.pth')

        if stopper(te_loss, model):
            print(f'Stopping early at epoch {epoch}.')
            model.load_state_dict(stopper.best_state) # rewind model to its state with lowest test loss
            stopped_early = True
            break

        # Save model in its final state
        torch.save({
        'epoch': epoch,
        'stopped_early': stopped_early,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'tr_loss': tr_loss,
        'te_loss': te_loss
        }, Path / f'final_save.pth')

    



# ------------- WORK IN PROGRESS -------------

def restart_training(configs):
    rundir = Path(configs.experiment_name)
    chkpts = list(rundir.glob('checkpoint-*.pth'))
    if len(chkpts) > 0:
        saved_epochs = [int(cpt.stem.split('-')[-1].split('.')[0]) for cpt in chkpts]
    pass

