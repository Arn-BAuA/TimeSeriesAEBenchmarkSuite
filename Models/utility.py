
import torch
from torch import nn
import numpy as np

activations = {
            "ReLU":torch.nn.ReLU,
            "Sigmoid":torch.nn.Sigmoid,
            "tanh":torch.nn.Tanh
        }

#Method for converting the arguments passed 
# in the Hyperparameters as string to the
#actual pytorch methods.
def strToActivation(string):
    return activations[string]

#Creates a Feed Forward Network, where the size of the
#layers change linear from input to output (e. g: If Input layer
#has 64 and output has 32 neurons and the network has 4 layers,
# the inner layers aree of size 56 and 48 neurons.)
def createLinearNetwork(InputSize,OutputSize,nLayers,activationFunction):
    
    if nLayers < 2:
        raise ValueError("The number of Layers for a linear Interpolated network must be at least 2 (now it's "+nLayers+")")

    #linar equation a lá y=mx+b to determine n neurons per layer (x = nInnerNeurons, y = neuronsPerLayer)
    # starts at x = 0 with nGlueLayerInput so some of the formulas are simpler as in the general case.
    x = np.arange(nLayers)
    m = (OutputSize-InputSize)/(nLayers-1)
    b = InputSize
            
    nNeurons = m*x + b

    layers = []

    for i,n in enumerate(nNeurons[:-1]):
        layers.append(torch.nn.Linear(int(n),int(nNeurons[i+1])))
        layers.append(activationFunction)

    return nn.Sequential(*layers)
    

