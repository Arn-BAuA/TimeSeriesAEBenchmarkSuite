import torch
from torch import nn
from BlockAndDatablock import block
from Models.utility import strToActivation,createLinearNetwork
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
                "HanningWindowBeforeFFT":True, #Uses the Hanning window before computing the FFT, if FFT encoder is there...
                "GlueLayerSize":2, #There is a stack of perceptrons that takes the output of the FFT- and TimeDecoder and Broadcasts them to the encoder input. That stack can have a height, specified here.
                "hasOnlyFFTEncoder":False,#if true, only a fft encoder is provided
                "ActivationFunction":"ReLU", #activation function used perceptrons across the net
               # "DownsampleByPooling":True, #if true, the downsampling in the decoder is done by pooling. if not, a layer of perceptrons does the job.
                "BatchNorm":True,#if true, a batchnorm is applied after each covolution.
            }
    
    #Little helper function for the creation of the encoders and the decoder
    #TODO THis looks somewhat similar to the function generating the FeedForwardAE. Maybe there is room for generalisation...
    def createLayers(self,layerScaleFactors):
        layers = []

        for i,factor in enumerate(layerScaleFactors[:-1]):
            nNeurons = int(np.ceil(self.HP["InputSize"]*factor))
            nNeuronsNext = int(np.ceil(self.HP["InputSize"]*layerScaleFactors[i+1])) #Neurons on the next Layer...

            #Convolution
            padding = 0
            dilation = 1
            kernel_size = self.HP["KernelSize"]
            stride = 1

            layers.append(nn.Conv1d(in_channels = self.Dimensions,
                                    out_channels = self.Dimensions,
                                    kernel_size = kernel_size,
                                    dilation = dilation,
                                    stride = stride,
                                    padding = padding,
                                    ))
            nConvOutput = int((nNeurons + (2*padding) -(dilation * (kernel_size -1)) -1)/stride)+1
            #Batch normal
            if(self.HP["BatchNorm"]):
                layers.append(nn.BatchNorm1d(num_features = self.Dimensions))
            #Up/Downscaling
            layers.append(torch.nn.Linear(nConvOutput,nNeuronsNext))
            layers.append(strToActivation(self.HP["ActivationFunction"])())
        

        return layers


    def __init__(self,Dimensions,device,**HyperParameters):

        self.device = device
        self.Dimensions = Dimensions

        block.__init__(self,"ConvolutionalAE",**HyperParameters)
        nn.Module.__init__(self)

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
            layers = self.createLayers([1]+self.HP["LayerSequenceEnc"]+[self.HP["LatentSize"]])
            self.TimeEncoder = nn.Sequential(*layers)
            self.TimeEncoder.to(device)

        if self.hasFFTEncoder:
            #Create FFT Encoder
            #TODO THis is the time domain input size at the begin of the input. It would be nice to make a version
            # WHere the input is actually oriented towards the fft output
            layers = self.createLayers([1]+self.HP["LayerSequenceEnc"]+[self.HP["LatentSize"]])
            self.FFTEncoder = nn.Sequential(*layers)
            self.FFTEncoder.to(device)
            if self.HP["HanningWindowBeforeFFT"]:
                self.hanningWindow = torch.hann_window(self.HP["InputSize"]).to(device)

        if self.hasGlueLayer:
            #Create Glue Layer
            nLatentNeurons = np.ceil(self.HP["InputSize"]*self.HP["LatentSize"])
            nGlueLayerInput = 2*nLatentNeurons
           
            if self.HP["GlueLayerSize"] < 2:
                self.HP["GlueLayerSize"] = 2
            
            glueLayerActivationFunction = strToActivation(self.HP["ActivationFunction"])()

            self.GlueLayer = createLinearNetwork(InputSize = nGlueLayerInput,
                                                 OutputSize = nLatentNeurons,
                                                 nLayers = self.HP["GlueLayerSize"],
                                                 activationFunction = glueLayerActivationFunction)
            
        #Create Decoder
        if self.hasGlueLayer:
            layers = self.createLayers([self.HP["LatentSize"]]+self.HP["LayerSequenceDec"]+[1])
        else:
            layers = self.createLayers(self.HP["LayerSequenceDec"]+[1])
            n1 = int(np.ceil(self.HP["LatentSize"]*self.HP["InputSize"]))
            n2 = int(np.ceil(self.HP["LayerSequenceDec"][0]*self.HP["InputSize"]))
            layers = [torch.nn.Linear(n1,n2)]+layers
            
        layers+=[torch.nn.Linear(self.HP["InputSize"],self.HP["InputSize"])]

        self.Decoder = nn.Sequential(*layers)

    def forward(self,x):
        
        if self.hasTimeEncoder:
            xTime = self.TimeEncoder(x)
        if self.hasFFTEncoder:
            if self.HP["HanningWindowBeforeFFT"]:
                fftInputX = x[:,:] * self.hanningWindow
            else:
                fftInputX = x
            xFFT = torch.fft.rfft(fftInputX)
            xFFT =xFFT.abs()
            xFFT = torch.cat([xFFT,torch.zeros([xFFT.size()[0],self.Dimensions,self.HP["InputSize"]-xFFT.size()[-1]]).to(self.device)],dim = -1)
            xFFT = self.FFTEncoder(xFFT)

        if self.hasGlueLayer:
            glueInput = torch.cat((xFFT,xTime),dim=-1)
            x = self.GlueLayer(glueInput)
        else:
            if self.hasTimeEncoder:
                x = xTime
            else:
                x = xFFT
        
        

        x = self.Decoder(x)
        return x


