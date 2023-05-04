import torch
from torch import nn
from BlockAndDatablock import block
from Models.utility import strToActivation

class Model(block,nn.Module): #Plain Feed Forward Encoder....
    
    def _getDefaultHPs(self):
        return {"InputSize":150,
                "LayerSequence":[100,50,20,50,100],
                "ActivationFunction":"ReLU"}
    
    def __init__(self,Dimensions,device,**HyperParameters):
        
        self.device = device
        self.Dimensions = Dimensions

        block.__init__(self,"FeedForwardAE",**HyperParameters) 
        nn.Module.__init__(self)
        
        
        LayerStack = [0]*(2*len(self.HP["LayerSequence"])+1)
        
        #Adding the input/output size to the front and end of the layer stack.
        actualSequence = [Dimensions*self.HP["InputSize"]]+self.HP["LayerSequence"]+[Dimensions*self.HP["InputSize"]]
        
        for i in range(0,len(actualSequence)-2):
            LayerStack[2*i] = torch.nn.Linear(actualSequence[i],actualSequence[i+1])
            LayerStack[2*i + 1] = strToActivation(self.HP["ActivationFunction"])()
        
        LayerStack[-1] = torch.nn.Linear(actualSequence[-2],actualSequence[-1])

        #Todo: Clean up and activationfunction as Hyperparameter
        self.model = nn.Sequential(*LayerStack)

        self.model.to(device)

    def forward(self,x):
        xShape = x.shape
        
        x = torch.flatten(x,start_dim=1)
        x = self.model(x)
        x = torch.reshape(x,xShape)

        return x


