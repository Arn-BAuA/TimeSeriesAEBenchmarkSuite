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

###Performance goals are performance goals in percents of the initial error, they are tracked
# for validation, training and test set

def benchmark(trainingSet,validationSet,testSet,model,trainer,n_epochs,pathToSave,device,SaveAfterEpochs = 10,PerformanceGoals = [70,50,20,15,10,5,3,1,0.5,0.1,0.01,0.001]):
    
    criterion = ...

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

    
    #########################################
    #   Begin of the Training...            #
    #########################################

    model.to(device)    
        
    history = {"train":[],"val":[]}
    
    performanceMetadata = {}
    
    
    dataSetTypes = ["Training Set","Validation Set","Test Set"]

    def getGoalName(goal,setType): #Goal names double as indices, hece the method
        str(goal)+"%-Goal on "+dataSetTypes[setType]

    for goal in performanceGoals:
        performaceMetadata[getGoalName(goal,0)] = "Not Reached" # Training Set
        performaceMetadata[getGoalName(goal,1)] = "Not Reached" # Validation Set
        performaceMetadata[getGoalName(goal,2)] = "Not Reached" # Test Set
    
    def calculateError(model,Dataset):
        Error = [0]*len(Dataset.Data())

        for i in range(0,len(Dataset.Data())):
            seq_true = Dataset.Data()[i]
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            Error.append(criterion(seq_true,seq_pred).item())
        
        return np.mean(Error) 


    def evaluatePerformance(model,DataSet,setType):
        

    for epoch in range(0,n_epochs):
        model,history = trainer.doEpoch(model,trainingSet,validationSet,history)
        print(f"Epoch: {epoch} , Train.Err.: {history['train'][-1]} , Val.Err.: {history['val'][-1]}")

        #check Performance goals,

        #if met or certain number of epochs:
        #export examples, save model stads

    trainer.finalizeTraining(model)

    #Save Trained Model
    #Save History

    #evaluate model

    #Save Performance Characteristics
    with open(resultFolder+"HyperParametersAndMetadata.json","w") as f:
        json.dump(runInformation,f,default=str,indent = 4)
