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

    def doEpoch(self,model,trainingSet,validationSet):
    
        model = model.train()
        for seq_true in trainingSet.Data():

            self.optimizer.zero_grad()
        
            seq_true = seq_true.to(self.device) # I think we can do this faster if we transfer the dataset to the gpu first and than do the training
            seq_pred = model(seq_true)
    
            loss = self.criterion(seq_pred, seq_true)

            loss.backward()
            self.optimizer.step()

        model = model.eval()
    
        return model

    def finalizeTraining(self,model):
        pass
