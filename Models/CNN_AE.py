import torch
from torch import nn
from BlockAndDatablock import block
from Models.utility import strToActivation

class Model(block,nn.Module): 
    

    #The idea of including the FFT encoder is inspired by a tutorial written by Ilia Zaitsev: 
    #https://www.kaggle.com/code/purplejester/pytorch-deep-time-series-classification
    # Layer structure is inspired by Tailai Wen and Roy Keyes (arxiv:1905.1362v1)
    def _getDefaultHPs(self):
        return {
                "InputSize":150,#size of the input data...
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
            
            self.TimeEncoder =

        if self.hasFFTEncoder:
            #Create FFT Encoder
            self.FFTEncoder = 

        if self.hasGlueLayer:
            #Create Glue Layer
            self.GlueLayer = 

        #Create Decoder

        self.Decoder = 

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

