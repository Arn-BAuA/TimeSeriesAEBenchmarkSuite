import torch
from torch import nn
from BlockAndDatablock import block

import globalArea

class Model(block,nn.Module): #Plain Feed Forward Encoder....
    
    def _getDefaultHPs(self):
        return {"InputSize":150}
    
    def __init__(self,Dimensions,device,**HyperParameters):
        
        self.device = device
        self.Dimensions = Dimensions

        block.__init__(self,**HyperParameters) 
        nn.Module.__init__(self)

        self.model = nn.Sequential(
                    torch.nn.Linear(Dimensions*self.HP["InputSize"] , 100),
                    torch.nn.ReLU(),
                    torch.nn.Linear(100 , 50),
                    torch.nn.ReLU(),
                    torch.nn.Linear(50 , 20),
                    torch.nn.ReLU(),
                    torch.nn.Linear(20 , 50),
                    torch.nn.ReLU(),
                    torch.nn.Linear(50 , 100),
                    torch.nn.ReLU(),    
                    torch.nn.Linear(100,Dimensions*self.HP["InputSize"])
                )

        self.model.to(device)

    def forward(self,x):
        x = torch.transpose(x,0,1)
        x = self.model(x)
        return torch.transpose(x,0,1)


