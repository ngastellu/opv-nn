import torch.nn  as nn


class Loss(nn.Module):
    def __init__(self,configs):
        super().__init__()
        supported_loss_funcs = ['mse', 'l1', 'huber']
        assert configs.loss_func in supported_loss_funcs, f"Loss function {configs.loss_func} not supported. Accepted loss functions are: {supported_loss_funcs}."
        if configs.loss_func == 'mse':
            self.loss_func = nn.MSELoss()
        elif configs.loss_func == 'l1':
            self.loss_func = nn.L1Loss()
        else:
            self.huber_delta = getattr(configs, 'huber_delta', 0.1)
            self.loss_func = nn.HuberLoss(delta=self.huber_delta)
    
    def forward(self,preds, targets):
        return self.loss_func(preds, targets)