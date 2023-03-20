##
# This training routine is "borrowed" from a tutorial on couriously.com
import torch
import numpy as np
from torch import nn
from torch import optim
import copy

import BlockAndDatablock.block

class Trainer(block):

    def __init__(self,model,device,**hyperParameters):

        block.__init__(device,hyperParameters)

        self.optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
        self.criterion = nn.L1Loss(reduction = "sum").to(device)

    
        best_model_wts = copy.deepcopy(model.state_dict())
        self.best_loss = 1e9

    def doEpoch(self,model,trainingSet,validationSet,history):
    
        train_loss = []
        for seq_true in trainingSet:

            self.optimizer.zero_grad()
        
            seq_true = seq_true.to(self.device) # I think we can do this faster if we transfer the dataset to the gpu first and than do the training
            seq_pred = model(seq_true)
    
            loss = self.criterion(seq_pred, seq_true)

            loss.backward()
            self.optimizer.step()

            train_loss.append(loss.item())

        val_loss = []

        model = model.eval()
    
        with torch.no_grad():
            for seq_true in validationSet:

                seq_true = seq_true.to(self.device)
                seq_pred = model(seq_true)

                loss = self.criterion(seq_pred,seq_true)
                val_loss.append(loss.item())

        train_loss = np.mean(train_loss)
        val_loss = np.mean(val_loss)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())

        return model,history

    def finalizeTraining(self,model):
        model.load_state_dict(best_model_wts)

