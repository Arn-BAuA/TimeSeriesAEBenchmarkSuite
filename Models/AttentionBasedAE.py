import torch
from torch import nn
from BlockAndDatablock import block
from Models.utility import strToActivation

class Model(block,nn.Module): #Using Vaswani Et al's Transformer as heart of an reconstruction based method...
    
    def _getDefaultHPs(self):
        return {"InputSize":150,
                "nWords":15,
                "WordSize":20,
                "nPreprocessorLayers":2,
                "ActivationFunctionPreprocessor":"ReLu",
                "nAttentionHeads":4,
                "nTrEncoderLayers":2,
                "nTrDecoderLayers":2,
                "nTrFFDim":40,
                "nPostprocessorLayers":2,
                "ActivationFunctionPostprocessor":"ReLu",}
    
    def __init__(self,Dimensions,device,**HyperParameters):
        
        self.device = device
        self.Dimensions = Dimensions

        block.__init__(self,"TransformerBasedModel",**HyperParameters) 
        nn.Module.__init__(self)
        

        

    def forward(self,x):
        

        return x


