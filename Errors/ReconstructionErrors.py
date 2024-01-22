import torch
import numpy as np

#A wrapper class that makes the default pytorch loss functions axessable in the Benchmark script.
class TorchErrorWrapper():

    def __init__(self,name,errorFunction,device):
        self.name = name
        self.device = device
        self.errorFunction = errorFunction.to(device)

    def Name(self):
        return self.name

    def calculate(self,model,DataPoint):
        Error = [0]*len(DataPoint)
        

        seq_true = DataPoint.to(self.device)
        seq_pred = model(seq_true)

        return self.errorFunction(seq_true,seq_pred).item()
        
        



