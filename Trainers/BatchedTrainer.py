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
                "TrainBatchSize":10,
                "nTrainBatches":100,
                "UseHistory":False,#If this is checked, after each epoch, the performance of the algorithm under test is measured on the validation set. If it supassed the best performance ever recorded in training, the model weights are saved. If it is worse than previous performance, the model weights are exchanged by the modle weights of the previous run. A little trick i picked up from this bolg: Time Series Anomaly Detection using LSTM Autoencoders with PyTorch in Python - https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/
                "ValidationBatchSize":10, #Will be ignored if use history is false
                "nValidationBatches":100, #Will be ignored if use history is false
                }

    def __init__(self,model,device,**hyperParameters):

        block.__init__(self,"BatchTrainer",**hyperParameters)
        
        self.device = device
        
        self.optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
        self.criterion = nn.L1Loss(reduction = "sum").to(device)

        if self.HPs["UseHistory"]:
            self.best_loss = float('inf')
    
    def __prepareBatches(self,nBatches,BatchSize,dataSet):
        
        data = dataSet.Data()
        Batches = [0] * nBatches
        
        for i in range(0,nBatches):
            toConcatenate = [0]*BatchSize

            for j in range(0,BatchSize):
                
                index=int(random()*len(data))
                toConcatenate[j] = data[index]

            Batches[i] = torch.cat(toConcatenate)
            Batches[i] = Batches[i].to(self.device)
        return Batches

    #Batching and Augmentation would be done here....
    def setDataSets(self,trainingSet,validationSet):
        self.trainBatches = self.__prepareBatches(self.HPs["nTrainBatches"],self.HPs["TrainBatchSize"],trainingSet)
        self.validationBatches = self.__prepareBatches(self.HPs["nValidationBatches"],self.HPs["ValidationBatchSize"],validationSet)
        

    def doEpoch(self,model):
    
        model = model.train()
        for seq_true in self.trainBatches:

            self.optimizer.zero_grad()
            
            seq_pred = model(seq_true)
            
            loss = self.criterion(seq_pred, seq_true)

            loss.backward()
            self.optimizer.step()

        model = model.eval()
        
        if self.HPs["UseHistory"]:
            val_loss = []
    
            with torch.no_grad():
                for seq_true in self.validationBatches:

                    seq_pred = model(seq_true)

                    loss = self.criterion(seq_pred,seq_true)
                    val_loss.append(loss.item())

            val_loss = np.mean(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_wts = copy.deepcopy(model.state_dict())


        return model

    def finalizeTraining(self,model):
        if self.HPs["UseHistory"]:
            if self.best_loss < float('inf'):
                #this means, that the model has been trained before, so that the self.best_model_wts exists
            model.load_state_dict(self.best_model_wts)
