import numpy as np


# import torch
import numpy as np
import csv
import os

class LoggerCSV:
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, epoch, counter, accuracy):
        try:
            if epoch == 0:
                mode = 'w'
                with open(f'outputs/logs/{self.model_name}_{counter}.csv', mode, newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Train loss", "Val Loss"])
            else:
                mode = 'a'
            
            with open(f'outputs/logs/{self.model_name}_{counter}.csv', mode, newline='') as file:
                writer = csv.writer(file)
                writer.writerow(accuracy)
            file.close()
        except FileNotFoundError:
            print("Wrong path to the log file.")

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, model_name, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        self.model_name = model_name
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, counter
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            np.save(
                f'outputs/models/{self.model_name}_{counter}',
                model
            )


class EarlyStopping:
    """Early stops the training if validation loss 
    doesn't improve after a given patience."""
    def __init__(
        self, model_name="model", patience=7
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7         
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_checkpoint = SaveBestModel(model_name)
    
    def __call__(
        self, val_loss, model, 
        epoch, count
    ):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, model, count)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, model, count)
            self.counter = 0
