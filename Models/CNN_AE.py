import torch
from torch import nn
from BlockAndDatablock import block
from Models.utility import strToActivation
import numpy as np

class Model(block,nn.Module): 
    

    #The idea of including the FFT encoder is inspired by a tutorial written by Ilia Zaitsev: 
    #https://www.kaggle.com/code/purplejester/pytorch-deep-time-series-classification
    # Layer structure is inspired by Tailai Wen and Roy Keyes (arxiv:1905.1362v1)
    def _getDefaultHPs(self):
        return {
                "InputSize":150,#size of the input data...
                "KernelSize":5,#Size of kernel for the convolution
                "LayerSequenceEnc":[0.7,0.5], #Size of the layers in relation to the input
                "LatentSize":0.2, #size of the latent space in relation to the input
                "LayerSequenceDec":[0.5,0.7], #size of the layer in relation to the input
                "hasFFTEncoder":False, #if true, a second encoder, beside the time encoder is added, that porcesses the fft. 
                "GlueLayerSize":2, #There is a stack of perceptrons that takes the output of the FFT- and TimeDecoder and Broadcasts them to the encoder input. That stack can have a height, specified here.
                "hasOnlyFFTEncoder":False,#if true, only a fft encoder is provided
                "ActivationFunction":"tanh", #activation function used perceptrons across the net
               # "DownsampleByPooling":True, #if true, the downsampling in the decoder is done by pooling. if not, a layer of perceptrons does the job.
                "BatchNorm":True,#if true, a batchnorm is applied after each covolution.
            }
    
    #Little helper function for the creation of the encoders and the decoder
    #TODO THis looks somewhat similar to the function generating the FeedForwardAE. Maybe there is room for generalisation...
    def createLayers(self,layerScaleFactors):
        layers = []

        for i,factor in enumerate(layerScaleFactors[:-1]):
            nNeurons = np.ceil(self.HP["InputSize"]*factor)
            nNeuronsNext = np.ceil(self.HP["inputSize"]*layerScaleFactors[i+1]) #Neurons on the next Layer...

            #Convolution
            layers.append(nn.Conv1d(in_channels = self.Dimensions,out_channels = self.Dimensions,kernel_size = self.HP["KernelSize"]))
            #Batch normal
            if(self.HP["BatchNorm"]):
                layers.append(nn.BatchNorm1d(num_features = self.Dimensions))
            #Downscaling
            layers.append(torch.nn.Linear(nNeurons,nNeuronsNext))
            layers.append(strToActivation(self.HP["ActivationFunction"]))
        
        return layers


    def __init__(self,Dimensions,device,**HyperParameters)

        self.device = device
        self.Dimensions = Dimensions

        block.__init__(self,"ConvolutionalAE",**HyperParameters)
        nn.Moduel.__init__(self)

        if self.HP["hasOnlyFFTEncoder"] == True:
            self.hasFFTEncoder = True
            self.hasGlueLayer = False
            self.hasTimeEncoder = False
        else:
            self.hasFFTEncoder = self.HP["hasFFTEncoder"]
            self.hasGlueLayer = self.HP["hasFFTEncoder"]
            self.hasTimeEncoder = True
        

        if self.hasTimeEncoder:
            #Create Time Encoder
            layers = createLayers(self.HP["LayerSequenceEnc"]+[self.HP["LatentSize"]])
            self.TimeEncoder = nn.Sequential(*layers)
            self.TimeEncoder.to(device)

        if self.hasFFTEncoder:
            #Create FFT Encoder
            layers = createLayers(self.HP["LayerSequenceEnc"]+[self.HP["LatentSize"]])
            layers = [torch.fft.fft]+layers
            self.FFTEncoder = nn.Sequential(*layers)
            self.FFTEncoder.to(device)

        if self.hasGlueLayer:
            #Create Glue Layer
            nLatentNeurons = np.ceil(self.HP["InputSize"]*self.HP["LatentSize"])
            nGlueLayerInput = 2*nLatentNeurons
           
            if self.HP["GlueLayerSize"] < 2:
                self.HP["GlueLayerSize"] = 2
            
            #linar equation a lá y=mx+b to determine n neurons per layer (x = nInnerNeurons, y = neuronsPerLayer)
            # starts at x = 0 with nGlueLayerInput so some of the formulas are simpler as in the general case.
            x = np.arange(self.HP["GlueLayerSize"])
            m = (nLatentNeurons-nGleuLayerInput)/(self.HP["GlueLayerSize"]-1)
            b = nGlueLayerInput
            
            nNeurons = m*x + b

            layers = []

            for i,n in enumerate(nNeurons[:-1]):
                layers.append(torch.nn.Linear(n,nNeurons[i+1]))
                layers.append(strToActivation(self.HP["ActivationFunction"]))

            self.GlueLayer = nn.Sequential(*layers)

        #Create Decoder
        if self.hasGlueLayer:
            layers = createLayers([self.HP["LatentSize"]]+self.HP["LayerSequenceDec"])
        else:
            layers = createLayers(self.HP["LayerSequenceDec"])
            n1 = np.ceil(self.HP["LatentSize"]*self.HP["InputSize"])
            n2 = np.ceil(self.HP["LayerSequenceDec"][0]*self.HP["InputSize"])
            layers = [torch.nn.Linear(n1,n2)]+layers

        self.Decoder = nn.Sequential(*layers)

    def forward(self,x):

        if self.hasTimeEncoder:
            xTime = self.TimeEncoder(x)
        if self.hasFFTEncoder:
            xFTT = self.FFTEncoder(x)

        if self.hasGlueLayer:
            glueInput = torch.cat((xFFT,xTime))
            x = self.GlueLayer(glueInput)
        else:
            if not self.hasTimeEncoder:
                x = xTime
            else:
                x = xFFT

        return self.Decoder(x)

