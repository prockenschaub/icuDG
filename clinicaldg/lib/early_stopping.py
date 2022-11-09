import torch

class EarlyStopping:
    # adapted from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    def __init__(self, patience=5, maximize=True):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.maximize = maximize

    def __call__(self, val_loss, step, state_dict, path): 
        score = -val_loss 

        if self.best_score is None:
            self.best_score = score
            self.step = step
            save_model(state_dict, path)
        elif (self.maximize and score <= self.best_score) or \
             (not self.maximize and score >= self.best_score):
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            save_model(state_dict, path)
            self.best_score = score
            self.step = step
            self.counter = 0

def save_model(state_dict, path):
    torch.save(state_dict, path)     