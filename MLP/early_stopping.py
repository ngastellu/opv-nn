class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None
        self.best_epoch = 0

    def __call__(self, val_loss, model):
        if self.patience > 0:
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
                self.best_state = model.state_dict()
            else:
                self.counter += 1

            return self.counter >= self.patience  # True = stop
        else:
            return False # don't ever stop early if self.patience = -1
