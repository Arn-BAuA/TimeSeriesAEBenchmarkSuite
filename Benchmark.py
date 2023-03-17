#!/bin/python

import torch

pathForResults = "Results/"


device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
else: 
    if torch.backends.mps.is_available():
        device = "mps"

def createResultDict(path):
    pass

def getDevice():
    return device

def benchmark(loadData,setWrapperHP,model,modelHP,trainerClass,trainerHP,n_Epochs,dimensions,pathToSave):
    
    #report the environemnt ()

    #write set HP
    #write model HP
    #write trainer HP

    trainingSet,validationSet,testSet = loadData(dimensions)
    #Calculate and save set characteristics here
        
    model.to(device)
        
    trainer = trainerClass(model,device,**trainerHP) 
        
    history = {"train":[],"val":[]}

    for epoch in range(0,n_Epochs):
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
