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
import copy
import shutil

def createFolderWithTimeStamp(folderName):
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
              Errors = [], #Can be normal or downstream errors
              SaveAfterEpochs = 10,
              inplace=True,
              n_exampleOutputsTraining   = [5,5,2],#NUmber of Examples that show [high Error, low Error,average Error]
              n_exampleOutputsValidation = [5,5,2],
              n_exampleOutputsTest       = [5,5,2],
              create_output = True, #for test purposes
              defaultError = TorchErrorWrapper("L1 Error",torch.nn.L1Loss(reduction = "sum"),device)):
    
    Errors = [defaultError] + Errors

    resultFolder = pathToSave
    if create_output:
        if inplace:
            resultFolder = pathForResults + resultFolder
            
            if os.path.exists(resultFolder):
                shutil.rmtree(resultFolder)

            os.mkdir(resultFolder)
        else:
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
    
    runInformation["Used Errors"] = []
    for err in Errors:
        runInformation["Used Errors"].append(err.Name())

    #TODO: AUtomatisch einen kleinen Steckbrief der Hardware mitloggen...
    hardwareInfo = {
                
                "UsedComputationDevice":device,
            }

    runInformation["Hardware Info"] = hardwareInfo

    gitHash = subprocess.check_output(["git","rev-parse","HEAD"]).decode("ascii").strip()
    runInformation["GITHash"] = gitHash
    
    

    #Setting up files and folders to write to in the training:
    MilestonePath = resultFolder+"Milestones" #Every n epochs, a snapshot of the model and some example evaluations are stored here
    
    if create_output:
        os.mkdir(MilestonePath)

    MilestonePath += "/"

    #Definition of the csv files
    CSVDelimiter = '\t' #so its more like tsv instead of csv...

    #Createing a Dict containning the Values for diffrent Errors on training and test set:

    EpochKey = "#Epoch"
    CPUTKey = "CPUTime in s"
    WallTKey = "WallTime in s"
    TSPostfixKey = " on Training Set"
    VSPostfixKey = " on Validation Set"
    TestPostfixKey = " on Test Set"
    ErrorKey = " Delta"

    ErrorColumnDict = {}
    
    ErrorColumnDict[EpochKey] = [0]*(n_epochs+1)

    for err in Errors:
        ErrorColumnDict[err.Name()+TSPostfixKey] = [0]*(n_epochs+1)
        ErrorColumnDict[err.Name()+VSPostfixKey] = [0]*(n_epochs+1)
        ErrorColumnDict[err.Name()+TestPostfixKey] = [0]*(n_epochs+1)
        ErrorColumnDict[err.Name()+TSPostfixKey+ErrorKey] = [0]*(n_epochs+1)
        ErrorColumnDict[err.Name()+VSPostfixKey+ErrorKey] = [0]*(n_epochs+1)
        ErrorColumnDict[err.Name()+TestPostfixKey+ErrorKey] = [0]*(n_epochs+1)
    
    ErrorColumnDict[CPUTKey] = [0]*(n_epochs+1)
    ErrorColumnDict[WallTKey] = [0]*(n_epochs+1)

    def writeToErrors(epoch,Errors,WallTime,CPUTime):    
        ErrorColumnDict[EpochKey][epoch] = epoch
        ErrorColumnDict[CPUTKey][epoch] = CPUTime
        ErrorColumnDict[WallTKey][epoch] = WallTime

        for key in Errors:
            ErrorColumnDict[key][epoch] = Errors[key]


    #########################################
    #   Begin of the Training...            #
    #########################################

    model.to(device)    
    
    #Calculate Error war hier definiert...

    TotalTrainingWallTime = 0
    TotalTrainingCPUTime = 0
    
    TrainingErrorOnFinalEpoch = [] #These are used to determine the examples that will be displayed..
    ValidationErrorOnFinalEpoch = [] 
    TestErrorOnFinalEpoch = [] 
    
    snapshotFolders = []
    snapshotModelWeights = []
    
    ErrorDistributionTS = pd.DataFrame()
    ErrorDistributionVS = pd.DataFrame()
    ErrorDistributionTest = pd.DataFrame()

    #Creates a folder with the modelweights and some example inputs and modeloutputs
    def createModelSnapshot(model,folderToSave,nameOfSnapshot):
        os.mkdir(folderToSave+nameOfSnapshot)
        snapshotDir = folderToSave+nameOfSnapshot+"/"
        torch.save(model.state_dict(),snapshotDir+"model.pth")
        
        return snapshotDir

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
            CPUTime = endCPUTime - startCPUTime
        else:
            WallTime = 0
            CPUTime = 0
        
        #################################
        ## # # Calculating the Errors
        ####

        EvaluatedErrors = {}

        for err in Errors:

            tsError = [0]*len(trainingSet.Data())
            for i,DataPoint in enumerate(trainingSet.Data()):
                tsError[i] = err.calculate(model,DataPoint)
            
            EvaluatedErrors[err.Name()+TSPostfixKey] = np.mean(tsError)
            EvaluatedErrors[err.Name()+TSPostfixKey+ErrorKey] = np.std(tsError)
            
            vsError = [0]*len(validationSet.Data())
            for i,DataPoint in enumerate(validationSet.Data()):
                vsError[i] = err.calculate(model,DataPoint)
            
            EvaluatedErrors[err.Name()+VSPostfixKey] = np.mean(vsError)
            EvaluatedErrors[err.Name()+VSPostfixKey+ErrorKey] = np.std(vsError)
            
            TestError = [0]*len(testSet.Data())
            for i,DataPoint in enumerate(testSet.Data()):
                TestError[i] = err.calculate(model,DataPoint)
            
            EvaluatedErrors[err.Name()+TestPostfixKey] = np.mean(TestError)
            EvaluatedErrors[err.Name()+TestPostfixKey+ErrorKey] = np.std(TestError)
            
            if epoch%SaveAfterEpochs == 0:
                ErrorDistributionTS["Epoch "+str(epoch)+" "+err.Name()] = tsError
                ErrorDistributionVS["Epoch "+str(epoch)+" "+err.Name()] = vsError
                ErrorDistributionTest["Epoch "+str(epoch)+" "+err.Name()] = TestError

            if err.Name() == defaultError.Name():
                if epoch == n_epochs:
                    #Final Epoch. The errors calculated here will be used to choose the examples for the snapshots...
                    TrainingErrorOnFinalEpoch = tsError
                    ValidationErrorOnFinalEpoch = vsError
                    TestErrorOnFinalEpoch = TestError

        
        TotalTrainingWallTime += WallTime
        TotalTrainingCPUTime += CPUTime

        #Errors[0] is always the default error
        print(f"Epoch: {epoch} , Train.Err.: {np.round(EvaluatedErrors[Errors[0].Name()+TSPostfixKey],2)} , Val.Err.: {np.round(EvaluatedErrors[Errors[0].Name()+VSPostfixKey],2)}")
        
        writeToErrors(epoch,EvaluatedErrors,TotalTrainingWallTime,TotalTrainingCPUTime)
        
        #if met or certain number of epochs
        #export examples, save model stads
        
        if epoch%SaveAfterEpochs == 0 and create_output:
            print(f"Logged Milestone for {epoch} epochs.")
            snapshotFolder = createModelSnapshot(model,MilestonePath,"Milestone at "+str(epoch)+" Epochs")
            snapshotFolders.append(snapshotFolder)
            snapshotModelWeights.append(copy.deepcopy(model.state_dict()))
            

    
    trainer.finalizeTraining(model)
    
    #Save Trained Model
    if create_output:
        snapshotFolder = createModelSnapshot(model,resultFolder,"Final Model")
        snapshotFolders.append(snapshotFolder)
        snapshotModelWeights.append(model.state_dict().copy())
        ErrorDistributionTS.to_csv(resultFolder+"TrainingSetDistribution.csv",sep="\t") 
        ErrorDistributionVS.to_csv(resultFolder+"ValidationSetDistribution.csv",sep="\t") 
        ErrorDistributionTest.to_csv(resultFolder+"TestSetDistribution.csv",sep="\t") 

    #Benchmark finished model:
    #What is the runtime for an average evaluation?

    ###################################
    # # ## #### # ### #
    # Creating the example evaluations
    #
    #Snapshot Examples are selected based on the performance of the model on the training, test and validation set.
 
    def saveExample(model,seq_true,pathToSave,labels = []):
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

        tr.reset_index(drop=True,inplace=True)    
        pr.reset_index(drop=True,inplace=True)    
        labels.reset_index(drop=True,inplace=True)    

        result = pd.concat([tr,labels.to_frame(),pr],axis=1)

        result.to_csv(pathToSave+".csv",sep = CSVDelimiter)

   
    def selectExamples(DataSet,Errors,nExampleMaxError,nExampleMinError,nExampleAVGError):
        #Idea: Select some examples with average errors (normal Performance)
        # and some with high or low errors
        
        errorIndicesSorted = np.argsort(Errors)
        indexLargest = errorIndicesSorted[-nExampleMaxError:]
        indexSmallest = errorIndicesSorted[:nExampleMinError]

        average = np.abs(Errors-np.mean(Errors))
        indexAverage = np.argsort(average)[:nExampleAVGError]

        return indexLargest,indexSmallest,indexAverage

    TSMaxErr,TSMinErr,TSAVGErr = selectExamples(trainingSet,TrainingErrorOnFinalEpoch,*n_exampleOutputsTraining)
    VSMaxErr,VSMinErr,VSAVGErr = selectExamples(validationSet,ValidationErrorOnFinalEpoch,*n_exampleOutputsValidation)
    TestMaxErr,TestMinErr,TestAVGErr = selectExamples(testSet,TestErrorOnFinalEpoch,*n_exampleOutputsTest)

    SelectedExamples = {
            "Training Set Example with High Error ":TSMaxErr,
            "Training Set Example with Low Error ":TSMinErr,
            "Training Set Example with Average Error ":TSAVGErr,
            "Validation Set Example with High Error ":VSMaxErr,
            "Validation Set Example with Low Error ":VSMinErr,
            "Validation Set Example with Average Error ":VSAVGErr,
            "Test Set Example with High Error ":TestMaxErr,
            "Test Set Example with Low Error ":TestMinErr,
            "Test Set Example with Average Error ":TestAVGErr,
            }

    runInformation["Selected Examples"] = SelectedExamples

    indexSets = [
            TSMaxErr,
            TSMinErr,
            TSAVGErr,
            VSMaxErr,
            VSMinErr,
            VSAVGErr,
            TestMaxErr,
            TestMinErr,
            TestAVGErr,
            ]
    fileNames = [
            "Training Set Example with High Error ",
            "Training Set Example with Low Error ",
            "Training Set Example with Average Error ",
            "Validation Set Example with High Error ",
            "Validation Set Example with Low Error ",
            "Validation Set Example with Average Error ",
            "Test Set Example with High Error ",
            "Test Set Example with Low Error ",
            "Test Set Example with Average Error ",
            ]
    dataSets = [
            trainingSet,
            trainingSet,
            trainingSet,
            validationSet,
            validationSet,
            validationSet,
            testSet,
            testSet,
            testSet,
            ]

    for i in range(0,len(snapshotFolders)):
        model.load_state_dict(snapshotModelWeights[i])
        
        for j,key in enumerate(SelectedExamples):
            for index in SelectedExamples[key]:
                if dataSets[j].hasLabels:
                    saveExample(model,dataSets[j].Data()[index],snapshotFolders[i]+key+"("+str(index)+")",dataSets[j].Labels()[index])
                else:
                    saveExample(model,dataSets[j].Data()[index],snapshotFolders[i]+key+"("+str(index)+")")

    #Save Performance Characteristics
    errorData = pd.DataFrame()

    for key in ErrorColumnDict:
        errorData[key] = ErrorColumnDict[key]
    
    if create_output:
        errorData.to_csv(resultFolder+"Errors.csv",sep=CSVDelimiter)

    #Save Metadata for this run
    if create_output:
        with open(resultFolder+"HyperParametersAndMetadata.json","w") as f:
            json.dump(runInformation,f,default=str,indent = 4)
    
    return resultFolder
