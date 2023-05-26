import torch
import numpy as np
from torch import nn
from BlockAndDatablock import block
from Models.utility import strToActivation,createLinearNetwork

class Model(block,nn.Module): #Using Vaswani Et al's Transformer as heart of an reconstruction based method...
    
    def _getDefaultHPs(self):
        return {"InputSize":150,
                "nWords":15,
                "WordSize":20,
                "nPreprocessorLayers":2,
                "ActivationFunctionPreprocessor":"Tanh",
                "FeedDirect":True,#if false, a linear feed forward network will be used instead of the embedding
                "nAttentionHeads":4,
                "nTrEncoderLayers":2,
                "nTrDecoderLayers":2,
                "TrFFDim":40,
                "nPostprocessorLayers":4,
                "ActivationFunctionPostprocessor":"Tanh",}
    
    def __init__(self,Dimensions,device,**HyperParameters):
        
        self.device = device
        self.Dimensions = Dimensions

        block.__init__(self,"TransformerBasedModel",**HyperParameters) 
        nn.Module.__init__(self)
        

        #Create Preprocessor
        self.beginOfWords =  np.floor(np.linspace(0,self.HP["InputSize"]-self.HP["WordSize"],self.HP["nWords"]))
        self.endOfWords = self.beginOfWords+self.HP["WordSize"]
        
        if not self.HP["FeedDirect"]:
            self.PreprocessorNetwork = createLinearNetwork(
                                                        InputSize = self.HP["WordSize"]*Dimensions,
                                                        OutputSize = self.HP["WordSize"],
                                                        nLayers = self.HP["nPreprocessorLayers"],
                                                        activationFunction = strToActivation(self.HP["ActivationFunctionPreprocessor"])() 
                                                        )
        
        if self.HP["FeedDirect"]:
            tInputSize = self.HP["WordSize"]*self.Dimensions
        else:
            tInputSize = self.HP["WordSize"]

        #Create Transformer
        self.Transformer = nn.Transformer(
                            d_model = tInputSize,
                            nhead = self.HP["nAttentionHeads"],
                            num_encoder_layers = self.HP["nTrEncoderLayers"],
                            num_decoder_layers = self.HP["nTrDecoderLayers"],
                            dim_feedforward = self.HP["TrFFDim"],
                            batch_first = True
                            )

        #Create Postprocessor 
        if self.HP["FeedDirect"]:
            ppInputSize = self.HP["WordSize"]*self.Dimensions*self.HP["nWords"]
        else:
            ppInputSize = self.HP["WordSize"]*self.HP["nWords"]
        
        self.PostprocessorNetwork = createLinearNetwork(
                                                        InputSize = ppInputSize,
                                                        OutputSize = self.HP["InputSize"]*Dimensions,
                                                        nLayers = self.HP["nPostprocessorLayers"],
                                                        activationFunction = strToActivation(self.HP["ActivationFunctionPostprocessor"])() 
                                                        )
        if not self.HP["FeedDirect"]:
            self.PreprocessorNetwork.to(device)
        
        self.Transformer.to(device)
        self.PostprocessorNetwork.to(device)

    def forward(self,x):
        
        #Preprocession
        embeddedWindows = [0]*self.HP["nWords"]

        for i in range(0,self.HP["nWords"]):
            window = x[:,:,int(self.beginOfWords[i]):int(self.endOfWords[i])]
            
            window = torch.flatten(window,start_dim=1)
            if self.HP["FeedDirect"]:
                embeddedWindows[i] = window 
            else:
                embeddedWindows[i] = self.PreprocessorNetwork(window)
        
        #Transformer Stage
        transformerEncoderInput = torch.stack(embeddedWindows,dim = 1)
        transformerDecoderInput = torch.zeros_like(transformerEncoderInput)

        transformerOutput = self.Transformer(transformerEncoderInput,transformerDecoderInput)

        #PostProcessing
        postprocessorInput = torch.flatten(transformerOutput,start_dim=1)
        output = self.PostprocessorNetwork(postprocessorInput)
        output = torch.reshape(output,x.shape)

        return output


