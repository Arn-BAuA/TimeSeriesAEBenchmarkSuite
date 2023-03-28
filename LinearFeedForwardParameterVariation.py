#!/bin/python

from AEModels.FeedForward import Model as ModelClass
from SetWrappers.AirQualityUCI import loadData
from Trainers.SingleInstanceTrainer import Trainer
from Benchmark import benchmark,initializeDevice,createFolderWithTimeStamp

device = initializeDevice()
Dimensions = 1 # Dataset dimensions

########################################
#### About this Experiment:
#
# This Experiment creates three fixed datasets.
# It than creates a variety of feed forward AEs
# to reconstruct the data.
# THe AEs all have in common, that the number of 
# nodes per layer is linear decreasing
# in the encoder and linear increasing
# in the decoder.
# For a fixed number of layers, the size of the 
# latent space gets altered

#Size of the sample window/ input of the encoder
WindowSize = 64

#The diffrent number of inner layers of the 
# encoder and decoder. THe shared latent space
# layer and the input and output layer are not counted
# in. A Autoencoder with 1 nLayer would have 5 actual layers
nLayers = [0,1,2,3]

#DIfferent number of nodes in the latent space.
nLatentSpaceNodes = [1,2,4,8,16,32,64]

#Takes nLayers and nLatentSpaceNodes and returns the Layers
#Hyperparameter for the feedforward ae class
def getLayerHP(enDecLayers,latentSpaceNodes):
    
    encoderLayers = [0]*enDecLayers
    decoderLayers = [0]*enDecLayers

    #number of layers in encoder/decoder including input and latent space
    n = enDecLayers+2
    
    #Linear Equation ax+b
    a = (float(latentSpaceNodes) - float(WindowSize))/float(n)
    b = float(WindowSize)

    for i in range(0,enDecLayers):
        nNodes = round(a*float(i+1) + b)
        encoderLayers[i] = nNodes
        decoderLayers[-1-i] = nNodes

    return encoderLayers + [latentSpaceNodes] +  decoderLayers

resultFolder = "Linear Parameter Variation "
resultFolder = createFolderWithTimeStamp(resultFolder)

trainingSet,validationSet,testSet = loadData(Dimensions,sampleWindowSize = WindowSize)

for nL in nLayers:
    for nN in nLatentSpaceNodes:
        innerLayers = getLayerHP(nL,nN)

        print()
        print("**************************************")
        print()
        print("Conducting Test for ",innerLayers,".")
        print()
        print("**************************************")
        print()

        model = ModelClass(Dimensions,device,LayerSequence=innerLayers,InputSize=WindowSize,ActivationFunction="tanh")
        trainer = Trainer(model,device)

        benchmark(trainingSet,
                  validationSet,
                  testSet,
                  model,
                  trainer,
                  n_epochs=100,
                  pathToSave=resultFolder+"/"+f"il_{nL}_lL_{nN}_Date_Time_",
                  device = device)
