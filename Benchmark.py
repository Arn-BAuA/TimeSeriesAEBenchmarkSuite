#!/bin/python

import torch


def initializeDevice():
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"
    else: 
        if torch.backends.mps.is_available():
            device = "mps"

    return device

pathForResults = "Results/"
import os
from datetime import datetime
import json
import pandas as pd

def benchmark(trainingSet,validationSet,testSet,model,trainer,n_epochs,pathToSave,device):
    
    now = str(datetime.now())
    resultFolder = pathForResults+pathToSave+now
    os.mkdir(resultFolder)
    
    resultFolder += "/"
    ####################################
    # Creating A Dictionary Containing all the rundefining Paramters
    # and save it as json
    ####################################

    runInformation = {}

    generalInformation = {
                "StreamDimension":trainingSet.Dimension(),
                "Used Dataset":trainingSet.Name(),
                "Used Trainer":trainer.Name(),
                "Used Model":model.Name(),
                "Number of epochs":n_epochs
            }
    
    runInformation["General Information"] = generalInformation
    
    runInformation["SetWrapper HPs"] = trainingSet.hyperParameters()
    runInformation["Model HPs"] = model.hyperParameters()
    runInformation["Trainer HPs"] = trainer.hyperParameters()

    #TODO: AUtomatisch einen kleinen Steckbrief der Hardware mitloggen...
    hardwareInfo = {
                "Used Device":device
            }

    runInformation["Hardware Info"] = hardwareInfo

    with open(resultFolder+"HyperParametersAndMetadata.json","w") as f:
        json.dump(runInformation,f,default=str,indent = 4)
    
    #########################################
    #   Begin of the Training...            #
    #########################################

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
