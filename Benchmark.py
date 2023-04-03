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
import time
import numpy as np
import subprocess

def createFolderWithTimeStamp(folderName,excludeResultFolder = False):
    now = str(datetime.now())
    os.mkdir(pathForResults+folderName+now)
    return folderName+now

from Errors.ReconstructionErrors import TorchErrorWrapper

###Performance goals are performance goals in percents of the initial error, they are tracked
# for validation, training and test set

device = initializeDevice()

def benchmark(trainingSet,validationSet,testSet,
              model,
              trainer,
              n_epochs,
              pathToSave,
              device,
              defaultError = TorchErrorWrapper("L1 Error",torch.nn.L1Loss(reduction = "sum"),device),
              Errors = [], #Can be normal or downstream errors
              SaveAfterEpochs = 10,
              n_exampleOutputs = 5):
    
    Errors = [defaultError] + Errors

    resultFolder = pathToSave
    resultFolder = pathForResults+createFolderWithTimeStamp(resultFolder)
    
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
                
                "UsedComputationDevice":device,
            }

    runInformation["Hardware Info"] = hardwareInfo

    gitHash = subprocess.check_output(["git","rev-parse","HEAD"]).decode("ascii").strip()
    runInformation["GITHash"] = gitHash
    
    

    #Save Metadata for this run
    with open(resultFolder+"HyperParametersAndMetadata.json","w") as f:
        json.dump(runInformation,f,default=str,indent = 4)
    
    #Setting up files and folders to write to in the training:
    MilestonePath = resultFolder+"Milestones" #Every n epochs, a snapshot of the model and some example evaluations are stored here
    GoalPath = resultFolder+"Goals" #Every time a performance Goal is reached, a snapshot of the model will be stored here

    os.mkdir(MilestonePath)
    os.mkdir(GoalPath)

    MilestonePath += "/"
    GoalPath += "/"

    #Definition of the csv files
    CSVDelimiter = '\t' #so its more like tsv instead of csv...

    #Createing a Dict containning the Values for diffrent Errors on training and test set:

    EpochKey = "#Epoch"
    CPUTKey = "CPUTime in s"
    WallTKey = "WallTime in s"
    TSPostfixKey = " on Training Set"
    VSPostfixKey = " on Validation Set"

    ErrorColumnDict = {}
    
    ErrorColumnDict[EpochKey] = [0]*(n_epochs+1)

    for err in Errors:
        ErrorColumnDict[err.Name()+TSPostfixKey] = [0]*(n_epochs+1)
        ErrorColumnDict[err.Name()+VSPostfixKey] = [0]*(n_epochs+1)
    
    ErrorColumnDict[CPUTKey] = [0]*(n_epochs+1)
    ErrorColumnDict[WallTKey] = [0]*(n_epochs+1)

    def writeToErrors(epoch,Errors,WallTime,CPUTime):    
        ErrorColumnDict[EpochKey][epoch] = epoch
        ErrorColumnDict[CPUTKey][epoch] = CPUTime
        ErrorColumnDict[WallTKey][epoch] = WallTime

        for key in Errors:
            ErrorColumnDict[key][epoch] = Errors[key]

    #Creates a folder with the modelweights and some example inputs and modeloutputs
    def createModelSnapshot(model,folderToSave,nameOfSnapshot):
        os.mkdir(folderToSave+nameOfSnapshot)
        snapshotDir = folderToSave+nameOfSnapshot+"/"

        torch.save(model.state_dict(),snapshotDir+"model.pth")
       
        def saveExample(seq_true,pathToSave):
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            
            seq_true = seq_true.to("cpu")
            seq_pred = seq_pred.to("cpu")
            

            tr = pd.DataFrame(seq_true.detach().numpy())
            pr = pd.DataFrame(seq_pred.detach().numpy())

            trColumnNames = []
            prColumnNames = []

            for i in range(0,trainingSet.Dimension()):
                trColumnNames.append("input_Dimension_"+str(i))
                prColumnNames.append("output_Dimension_"+str(i))
            
            tr=tr.set_axis(trColumnNames,axis=1)
            pr=pr.set_axis(prColumnNames,axis=1)

            
            result = pd.concat([tr,pr],axis=1)
            result.to_csv(pathToSave+".csv",sep = CSVDelimiter)

        for i in range(0,n_exampleOutputs):
            saveExample(trainingSet.Data()[i],snapshotDir+"TrainingSetExample"+str(i+1))
            saveExample(validationSet.Data()[i],snapshotDir+"ValidationSetExample"+str(i+1))
            saveExample(testSet.Data()[i],snapshotDir+"TestSetExample"+str(i+1))

    #########################################
    #   Begin of the Training...            #
    #########################################

    model.to(device)    
    
    #Calculate Error war hier definiert...

    TotalTrainingWallTime = 0
    TotalTrainingCPUTime = 0

    for epoch in range(0,n_epochs+1):
        
        if not epoch == 0:#No Training at the start for the initial impression
            # ### ## # ## ######### ## # ### ### ## ##
            #### # ##### ## ## # ###### ## #### ## ## # ###
            ###### ######### # ##### ## ### #

            startTime = time.time()
            startCPUTime = time.process_time()

            model = trainer.doEpoch(model,trainingSet,validationSet)

            endTime = time.time()
            endCPUTime = time.process_time()
        
            # ### ## # ## ######### ## # ### ### ## ##
            #### # ##### ## ## # ###### ## #### ## ## # ###
            ###### ######### # ##### ## ### #
            WallTime = endTime - startTime
            CPUTime = endCPUTime - startTime
        else:
            WallTime = 0
            CPUTime = 0

        EvaluatedErrors = {}

        for err in Errors:
            EvaluatedErrors[err.Name()+TSPostfixKey] = err.calculate(model, trainingSet)
            EvaluatedErrors[err.Name()+VSPostfixKey] = err.calculate(model, validationSet) 
        
        TotalTrainingWallTime += WallTime
        TotalTrainingCPUTime += CPUTime

        #Errors[0] is always the default error
        print(f"Epoch: {epoch+1} , Train.Err.: {EvaluatedErrors[Errors[0].Name()+TSPostfixKey]} , Val.Err.: {EvaluatedErrors[Errors[0].Name()+VSPostfixKey]}")
        
        writeToErrors(epoch,EvaluatedErrors,TotalTrainingWallTime,TotalTrainingCPUTime)
        
        #if met or certain number of epochs
        #export examples, save model stads
        
        if epoch%SaveAfterEpochs == 0:
            print(f"Logged Milestone for {epoch} epochs.")
            createModelSnapshot(model,MilestonePath,"Milestone at "+str(epoch)+" Epochs")

    trainer.finalizeTraining(model)
    
    #Save Trained Model
    createModelSnapshot(model,resultFolder,"Final Model")
    

    #Benchmark finished model:
    #What is the runtime for an average evaluation?


    #Save Performance Characteristics
    errorData = pd.DataFrame()

    for key in ErrorColumnDict:
        errorData[key] = ErrorColumnDict[key]
    
    errorData.to_csv(resultFolder+"Errors.csv",sep=CSVDelimiter)

    return resultFolder
