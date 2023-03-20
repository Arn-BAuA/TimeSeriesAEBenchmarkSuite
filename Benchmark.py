#!/bin/python

import torch

pathForResults = "Results/"

def createResultDict(path):
    pass

def initializeDevice():
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"
    else: 
        if torch.backends.mps.is_available():
            device = "mps"

    return device

def benchmark(trainingSet,validationSet,testSet,model,trainer,n_epochs,pathToSave,device):
    
    #report the environemnt ()

    #write set HP
    #write model HP
    #write trainer HP
    #Calculate and save set characteristics here
        
    model.to(device)    
        
    history = {"train":[],"val":[]}

    for epoch in range(0,n_epochs):
        model,history = trainer.doEpoch(model,trainingSet,validationSet,history)
        print(f"Epoch: {epoch} , Train.Err.: {history['train'][-1]} , Val.Err.: {history['val'][-1]}")

        #chech Performance goals,

        #if met or certain number of epochs:
        #export examples, save model stads

    trainer.finalizeTraining(model)

    #Save Trained Model
    #Save History

    #evaluate model

    #Save Performance Characteristics
