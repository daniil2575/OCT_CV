import torch, os

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.count = 0
        self.should_stop = False
    def step(self, metric):
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric; self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.should_stop = True

class Checkpoint:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
    def save(self, model, cfg):
        torch.save({'model': model.state_dict(), 'cfg': cfg}, self.path)
