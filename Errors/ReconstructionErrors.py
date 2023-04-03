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

    def calculate(self,model,Dataset):
        Error = [0]*len(Dataset.Data())

        for i in range(0,len(Dataset.Data())):
            seq_true = Dataset.Data()[i]
            seq_true = seq_true.to(self.device)
            seq_pred = model(seq_true)

            Error.append(self.errorFunction(seq_true,seq_pred).item())
        
        return np.mean(Error) 



