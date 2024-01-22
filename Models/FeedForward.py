
import numpy as np
import torch
from torch import nn
from BlockAndDatablock import block
from Models.utility import strToActivation

class Model(block,nn.Module): #Plain Feed Forward Encoder....
    
    def _getDefaultHPs(self):
        return {"InputSize":150,
                "LayerSequence":[1,0.7,0.3,0.2,0.3,0.7,1],#in multiples of the input ... first and last must be 0
                "ActivationFunction":"ReLU",
                "SlicingBeforeFlatten":True,#If Slicing is true, the tensors will not be flattend, the tensors will be sliced along the time axis, these slices will than be flattend and concatenated.
                "SliceLength":0.1, #of sequence lenght
                "LayerSequenceLinear":True, #if this is true, the layer sequence above will be ignored. instead a linear progression of th layer size to the latent space will be calculated. The layer Sequence will than be overwritten by that.
                "LatentSpaceSize":0.1,
                "NumLayersPerPart":3
                }
    
    def __init__(self,Dimensions,device,**HyperParameters):
        
        self.device = device
        self.Dimensions = Dimensions

        block.__init__(self,"FeedForwardAE",**HyperParameters) 
        nn.Module.__init__(self)

        inputLen = Dimensions*self.HP["InputSize"]
        self.inputLen = inputLen

        if self.HP["SlicingBeforeFlatten"]:
            nWindows = int(np.floor(1/self.HP["SliceLength"])+1)
            windowLength = np.ones(nWindows-1)*np.floor(self.HP["SliceLength"]*self.inputLen)
            windowLength = np.concatenate([[0],windowLength,[self.inputLen-np.sum(windowLength)]])
            windowEndpoints = np.tril(np.ones(len(windowLength))).dot(windowLength)
            self.windowEndpoints = windowEndpoints.astype(int)

        if self.HP["LayerSequenceLinear"]:
            Layers = np.linspace(1,self.HP["LatentSpaceSize"],self.HP["NumLayersPerPart"])
            Layers = np.concatenate([Layers,np.flip(Layers[:-1])])
            self.HP["LayerSequence"] = Layers
        
        if len(self.HP["LayerSequence"] == 1):
            LayerStack = [0]*(3)
            
            numNeuronsMiddle = int(np.ceil(self.HP["LayerSequence"][0]*inputLen))

            LayerStack[0] = torch.nn.Linear(inputLen,numNeuronsMiddle)
            LayerStack[1] = strToActivation(self.HP["ActivationFunction"])()
            LayerStack[2] = torch.nn.Linear(numNeuronsMiddle,inputLen)
        

        else:
            LayerStack = [0]*(2*len(self.HP["LayerSequence"])-3)
        
            #Adding the input/output size to the front and end of the layer stack.
            actualSequence = np.ceil(self.HP["LayerSequence"]*inputLen).astype(int)
       
            for i in range(0,len(actualSequence)-2):

                nLastLayer = actualSequence[i]
                if nLastLayer == 0:
                    nLastLayer = 1
            
                nThisLayer = actualSequence[i+1]
                if nThisLayer == 0:
                    nThisLayer = 1

                LayerStack[2*i] = torch.nn.Linear(nLastLayer,nThisLayer)
                LayerStack[2*i + 1] = strToActivation(self.HP["ActivationFunction"])()
        
            LayerStack[-1] = torch.nn.Linear(actualSequence[-2],actualSequence[-1])
        
        #Todo: Clean up and activationfunction as Hyperparameter
        self.model = nn.Sequential(*LayerStack)

        self.model.to(device)

    def forward(self,x):
        xShape = x.shape

        if self.HP["SlicingBeforeFlatten"]:
            snippets = [0]*(len(self.windowEndpoints)-1)
            snippetShape = [0]*len(snippets)

            for i in range(1,len(self.windowEndpoints)):
                snippets[i-1] = x[:,:,self.windowEndpoints[i-1]:self.windowEndpoints[i]]
                snippetShape[i-1] = snippets[i-1].shape
                snippets[i-1] = torch.flatten(snippets[i-1],start_dim=1)

            x = torch.cat(snippets,-1)
                
        else:
            x = torch.flatten(x,start_dim=1)

        
        x = self.model(x)
        
        if self.HP["SlicingBeforeFlatten"]:
            cutPoints = self.windowEndpoints * self.Dimensions
            reshapedSnippets = [0]*(len(cutPoints)-1)
            
            for i in range(1,len(cutPoints)):
                reshapedSnippets[i-1] = torch.reshape(x[:,cutPoints[i-1]:cutPoints[i]],snippetShape[i-1])
            
            x = torch.cat(reshapedSnippets,-1)
        else:
            x = torch.reshape(x,xShape)
        
        return x


