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

###Performance goals are performance goals in percents of the initial error, they are tracked
# for validation, training and test set

def benchmark(trainingSet,validationSet,testSet,model,trainer,n_epochs,pathToSave,device,SaveAfterEpochs = 10,PerformanceGoals = [70,50,20,15,10,5,3,1,0.5,0.1,0.01,0.001],n_exampleOutputs = 5):
    
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

    #This file contains the error on trainingset and validationset and the runtime of the training
    ErrorsFile = open("Errors.csv","w")
    ErrorsFile.write("Epoch"+CSVDelimiter+"Training Set Error"+CSVDelimiter+"Validation Set Error"+CSVDelimiter+"Wall Time of Epoch (s)"+CSVDelimiter+"CPU Time of Epoch (s)"+"\n")

    def writeToErrors(epoch,TSError,VSError,WallTime,CPUTime):
        ErrorsFile.write(str(epoch)+CSVDelimiter+str(TSError)+CSVDelimiter+str(VSError)+CSVDelimiter+str(WallTime)+CSVDelimiter+str(CPUTime)+"\n")

    #This file v
    TrainingSetGoals = open("TSGoals","w")
    TrainingSetGoals.write("Goal Target (%)"+CSVDelimiter+"Reached in Epoch"+CSVDelimiter+"Elapsed Training Wall Time (s)"+ CSVDelimiter+"Elapsed Training CPU Time (s)"+"\n")

    ValidationSetGoals = open("VSGoals","w")
    ValidationSetGoals.write("Goal Target (%)"+CSVDelimiter+"Reached in Epoch"+CSVDelimiter+"Elapsed Training Wall Time (s)"+ CSVDelimiter+"Elapsed Training CPU Time (s)"+"\n")

    def writeToGoals(tableFile,goal,epoch,totalWallTime,totalCPUTime):
        tableFile.write(str(goal)+CSVDelimiter+str(epoch)+CSVDelimiter+str(totalWallTime)+CSVDelimiter+str(totalCPUTime)+"\n")

    #Creates a folder with the modelweights and some example inputs and modeloutputs
    def createModelSnapshot(model,folderToSave,nameOfSnapshot):
        os.mkdir(folderToSave+nameOfSnapshot)
        snapshotDir = folderToSave+nameOfSnapshot+"/"

        torch.save(model.state_dict(),snapshotDir+"model.pth")
       
        def saveExample(seq_true,pathToSave):
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            #seq_pred = seq_pred.to("cpu")

            tr = pd.DataFrame(seq_true.numpy())
            pr = pd.DataFrame(seq_pred.numpy())

            trColumnNames = []
            prColumnNames = []

            for i in range(0,trainingSet.Dimension):
                trColumnNames.append("input_Dimension_"+str(i))
                prColumnNames.append("output_Dimension_"+str(i))
            
            tr.set_axis(trColumnNames,axis=1)
            pr.set_axis(prColumnNames,axis=1)

            result = pd.concat([tr,pr],axis=1)
            result.to_csv(pathToSave+".csv",sep = CSVDelimiter)

        for i in range(0,n_exampleOutputs):
            saveExample(trainingSet.Data[i],snapshotDir+"TrainingSetExample"+str(i+1))
            saveExample(validationSet.Data[i],snapshotDir+"ValidationSetExample"+str(i+1))
            saveExample(testSet.Data[i],snapshotDir+"TestSetExample"+str(i+1))

    #########################################
    #   Begin of the Training...            #
    #########################################
    
    unreachedTSGoals = PerformanceGoals.copy()
    unreachedVSGoals = PerformanceGoals.copy()

    model.to(device)    

    def calculateError(model,Dataset):
        Error = [0]*len(Dataset.Data())

        for i in range(0,len(Dataset.Data())):
            seq_true = Dataset.Data()[i]
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            Error.append(criterion(seq_true,seq_pred).item())
        
        return np.mean(Error) 

    TotalTrainingWallTime = 0
    TotalTrainingCPUTime = 0

    for epoch in range(0,n_epochs):
        
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

        TrainingError = calculateError(model, trainingSet)
        ValidationError = calculateError(model, validationSet) 
        
        WallTime = endTime - startTime
        CPUTime = endCPUTime - startTime

        TotalTrainingWallTime += WallTime
        TotalTrainingCPUTime += CPUTime

        print(f"Epoch: {epoch} , Train.Err.: {TrainingError} , Val.Err.: {ValidationError}")
        
        writeToErrors(epoch,TrainingError,ValidationError,WallTime,CPUTime)
        
        #if met or certain number of epochs
        #export examples, save model stads

        #check Performance goals,
        
        def checkAndUpdateGoals(Error,initialError,unreachedGoals,fileToWriteResult):
            ErrPercent = 100*(Error / initialError)

            for i in range(0,len(unreachedGoals)):

                if unreachedGoals < ErrPercent:
                    #The Goal with index i is the first of the unreached that is not reached in this run
                    # The others which are befor it in the list are reached (There is the possibility to
                    # Reach multible performance goals in one epoch)
                    for j in range(0,i):
                        writeToGoals(fileToWriteResult,unreachedGoals[j],epoch,TotalTrainingWallTime,TotalTrainingCPUTime)

                    createModelSnapshot(model,GoalPath,"Goals Reached at Epoch "+str(epoch)):

                    unreachedGoals = unreachedGoals[i:]

                    break;

            return unreachedGoals

        if epoch == 0:
            #The specified Performance Goals are in percent of the initial error.
            #To evaluate this percentage, the initial error is needed.
            initialTrainingError = TrainingError
            initialValidationError = ValidationError
        else:
            unreachedTSGoals = checkAndUpdateGoals(TrainingError,initialTrainingError,unreachedTSGoals,trainingSetGoals)
            unreachedVSGoals = checkAndUpdateGoals(ValidationError,initialValidationError,unreachedVSGoals,validationSetGoals)
        
        if epoch%SaveAfterEpochs == 0:
            createModelSnapshot(model,MilestonePath,"Milestone at "+str(epoch)+" Epochs"):

    trainer.finalizeTraining(model)
    
    #Save Trained Model
    createModelSnapshot(model,resultFolder,"Final Model")
    

    #Benchmark finished model:
    #What is the runtime for an average evaluation?


    #Save Performance Characteristics
    ErrorsFile.close()

    for goal in unreachedTSGoals:
        writeToGoals(TrainingSetGoals,goal,"NaN")
    
    TrainingSetGoals.close()

    for goal in unreachedTVSGoals:
        writeToGoals(ValidationSetGoals,goal,"NaN")
    
    ValidationSetGoals.close()
