##
# This training routine is "borrowed" from a tutorial on couriously.com
import torch
import numpy as np
from torch import nn
from torch import optim
import copy

from BlockAndDatablock import block

class Trainer(block):

    def __init__(self,model,device,**hyperParameters):

        block.__init__(self,"SingleInstanceTrainer",**hyperParameters)
        
        self.device = device
        
        self.optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
        self.criterion = nn.L1Loss(reduction = "sum").to(device)

    
        best_model_wts = copy.deepcopy(model.state_dict())
        self.best_loss = 1e9

    def setDataSets(self,trainingSet,validationSet):
        self.trainingSet = trainingSet
        self.validationSet = validationSet

    def doEpoch(self,model,trainingSet,validationSet):
    
        for seq_true in trainingSet.Data():

            self.optimizer.zero_grad()
        
            seq_true = self.seq_true.to(self.device) # I think we can do this faster if we transfer the dataset to the gpu first and than do the training
            seq_pred = model(seq_true)
    
            loss = self.criterion(seq_pred, seq_true)

            loss.backward()
            self.optimizer.step()

        val_loss = []

        model = model.eval()
    
        with torch.no_grad():
            for seq_true in self.validationSet.Data():

                seq_true = seq_true.to(self.device)
                seq_pred = model(seq_true)

                loss = self.criterion(seq_pred,seq_true)
                val_loss.append(loss.item())

        val_loss = np.mean(val_loss)

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())

        return model

    def finalizeTraining(self,model):
        model.load_state_dict(self.best_model_wts)

