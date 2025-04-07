import torch


class EarlyStopping:
    def __init__(self, patience, min_delta, identifier):
        self.patience = patience
        self.min_delta = min_delta
        self.best_result = float('inf')
        self.counter = 0
        self.save_path = f'Models/{identifier}.pt'

    def __call__(self, current_result, model):
        if current_result < self.best_result - self.min_delta:
            self.best_loss = current_result
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1

        return self.counter >= self.patience
