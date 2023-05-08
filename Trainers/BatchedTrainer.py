##
# This training routine is "borrowed" from a tutorial on couriously.com
import torch
import numpy as np
from torch import nn
from torch import optim
import copy
from random import random

from BlockAndDatablock import block

class Trainer(block):
    
    
    def _getDefaultHPs(self):
        return {
                "BatchSize":10,
                "nBatches":100,
                }

    def __init__(self,model,device,**hyperParameters):

        block.__init__(self,"BatchTrainer",**hyperParameters)
        
        self.device = device
        
        self.optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
        self.criterion = nn.L1Loss(reduction = "sum").to(device)
    
    #Batching and Augmentation would be done here....
    def setDataSets(self,trainingSet,validationSet):
        data = trainingSet.Data()
        Batches = [0] * self.HP["nBatches"]
        
        for i in range(0,self.HP["nBatches"]):
            toConcatenate = [0]*self.HP["BatchSize"]

            for j in range(0,self.HP["BatchSize"]):
                
                index=int(random()*len(data))
                toConcatenate[j] = data[index]

            Batches[i] = torch.cat(toConcatenate)
        
        print(Batches[0].size())
        self.Batches = Batches

    def doEpoch(self,model):
    
        model = model.train()
        for seq_true in self.Batches:

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
