# Set Wraper for any of the datasets in the UCR Archive (https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
# The 

import pandas as pd
from random import random
import numpy as np
import torch
from BlockAndDatablock import DataBlock
from Utility.DataSetSampler import fromClassification2AnoDetect as sampleDataSet
from Utility.DataSetSampler import splitDataframeAlongRows,selectByLabel

import os
dirname = os.path.dirname(__file__)
ECGPath = os.path.join(dirname,"../data/ECG/")


def loadData(dimensions,**hyperParameters):

    defaultHyperParameters = {
            "ArrythmiaNormals":[0],#Classes from arrythmia that are used as normals, none, if left emty
            "ArrythmiaAnomalys":[1,2,3,4],#classes from arrythmia that are used as anomalies, none if left empty
            "PTBNormals":[],#Classes from PTB that are concidered as normal, none if left empty
            "PTBAnomalys":[],#Classes from the PTB that are used as anomalies, none if left empty
            "AnomalyPercentageTrain":0,#Percentage of anomalies in training set
            "AnomalyPercentageValidation":10, #percentage of anomalies in validation set
            "AnomalyPercentageTest":10,# percentage of anomalies in test set
            "SameClassForAllDimensions":False,#when true: All dimensions are of the same class, else, they are random
            "AllDimensionsAnomal":False,#If this is true and the value is an anomaly, all the dimensions are an anomaly
            "nAnomalDimensions":1,#if it is an anomaly: How many dimensions are anomal
            "UseArythmiaSplit":True,#For the used arythmia data: Use the split for training and test like in the original csv files:
            "PercentTrainingSet":70,# The percentage of samples that is split off for the training.
            "PercentValidationSet":20, # The percentage of samples that is split off for the validation.
            "PercentTestSet":10, # The percentage of samples that is split off for th testing.
            "TrainingSetSize":400,
            "ValidationSetSize":100,
            "TestSetSize":30,
        }
    HPs={**defaultHyperParameters,**hyperParameters}
    

    #Loading sets (These Dataframes are ment to be representations of the files in the class.)
    arrythmiaTrain = pd.read_csv(ECGPath+"mitbih_train.csv",sep=',',header=None)
    arrythmiaTest = pd.read_csv(ECGPath+"mitbih_test.csv",sep=',',header=None)
    
    #sorting out Arythmia classes that are not used.
    classLabelsTotal = [0,1,2,3,4]
    classLabelsUsed = HPs["ArrythmiaNormals"] + HPs["ArrythmiaAnomalys"]
    
    labelColumn = arrythmiaTrain.columns[-1] 
    
    for label in classLabelsTotal:
        if not label in classLabelsUsed:
            #label is not used
            arrythmiaTrain.drop(arrythmiaTrain[arrythmiaTrain[labelColumn]==label].index)
            arrythmiaTest.drop(arrythmiaTest[arrythmiaTest[labelColumn]==label].index)
    
    if HPs["UseArythmiaSplit"]:
        totalPercentage = float(HPs["PercentTrainingSet"]+HPs["PercentValidationSet"])
        trainFactor = float(HPs["PercentTrainingSet"])/totalPercentage
        validationFactor = float(HPs["PercentValidationSet"])/totalPercentage
        arrythmiaTrain,arrythmiaValidation = splitDataframeAlongRows([trainFactor,validationFactor],arrythmiaTrain)
    else:
        totalPercentage = float(HPs["PercentTrainingSet"]+HPs["PercentValidationSet"]+HPs["PercentageTestSet"])
        trainFactor = float(HPs["PercentTrainingSet"])/totalPercentage
        validationFactor = float(HPs["PercentValidationSet"])/totalPercentage
        testFactor = float(HPs["PercentTestSet"])/totalPercentage
        
        arrythmiaTrain,arrythmiaValidation,arrythmiaTest = splitDataframeAlongRows([trainFactor,validationFactor,testFactor],pd.concat(arrythmiaTrain,arrythmiaTest))
    
    #Distributing Arythmia
    trainingDataNormal = selectByLabel(arrythmiaTrain,HPs["ArrythmiaNormals"],labelColumn)
    trainingDataAnomal = selectByLabel(arrythmiaTrain,HPs["ArrythmiaAnomalys"],labelColumn)
    validationDataNormal = selectByLabel(arrythmiaValidation,HPs["ArrythmiaNormals"],labelColumn)
    validationDataAnomal = selectByLabel(arrythmiaValidation,HPs["ArrythmiaAnomalys"],labelColumn)
    testDataNormal = selectByLabel(arrythmiaTest,HPs["ArrythmiaNormals"],labelColumn)
    testDataAnomal = selectByLabel(arrythmiaTest,HPs["ArrythmiaAnomalys"],labelColumn)


    #Loading the PTB Sets if Necessairy.
    PTBNormalsLoaded = False
    PTBAnomalysLoaded = False

    if not len(HPs["PTBNormals"]) == 0:
        if HPs["PTBNormals"] == 1:
            PTBNormals = pd.read_csv(ECGPath+"ptbdb_anormal.csv",sep=',',header=None)
        else:
            PTBNormals = pd.read_csv(ECGPath+"ptbdb_normal.csv",sep=',',header=None)
        PTBNormalsLoaded = True
    if not len(HPs["PTBAnomalys"]) == 0:
        if HPs["PTBAnomalys"] == 1:
            PTBAnomals = pd.read_csv(ECGPath+"ptbdb_anormal.csv",sep=',',header=None)
        else:
            PTBAnomals = pd.read_csv(ECGPath+"ptbdb_normal.csv",sep=',',header=None)
        PTBAnomalysLoaded = True
    
    if PTBAnomalysLoaded or PTBNormalsLoaded:
        #Distributing the PTB-Sets
        totalPercentage = float(HPs["PercentTrainingSet"]+HPs["PercentValidationSet"]+HPs["PercentTestSet"])
        trainFactor = float(HPs["PercentTrainingSet"])/totalPercentage
        validationFactor = float(HPs["PercentValidationSet"])/totalPercentage
        testFactor = float(HPs["PercentTestSet"])/totalPercentage
   
        percentages = [trainFactor,validationFactor,testFactor]

    if PTBNormalsLoaded:
        PTBTrainDataNormal,PTBValidationDataNormal,PTBTestDataNormal= splitDataframeAlongRows(percentages,PTBNormals)
        
        trainingDataNormal = pd.concat([PTBTrainDataNormal,trainingDataNormal]) 
        validationDataNormal = pd.concat([PTBValidationDataNormal,validationDataNormal])
        testDataNormal = pd.concat([PTBTestDataNormal,testDataNormal])
    
    if PTBAnomalysLoaded:
        PTBTrainDataAnomal,PTBValidationDataAnomal,PTBTestDataAnomal= splitDataframeAlongRows(percentages,PTBAnomals)
        
        trainingDataAnomal = pd.concat([PTBTrainDataAnomal,trainingDataAnomal]) 
        validationDataAnomal = pd.concat([PTBValidationDataAnomal,validationDataAnomal])
        testDataAnomal = pd.concat([PTBTestDataAnomal,testDataAnomal])


    #droping the first column with the labels
    trainingDataNormal = trainingDataNormal.drop(labelColumn,axis=1)
    trainingDataAnomal = trainingDataAnomal.drop(labelColumn,axis=1)
    validationDataNormal = validationDataNormal.drop(labelColumn,axis=1)
    validationDataAnomal = validationDataAnomal.drop(labelColumn,axis=1)
    testDataNormal = testDataNormal.drop(labelColumn,axis=1)
    testDataAnomal = testDataAnomal.drop(labelColumn,axis=1)

    trainingSet = [0]*HPs["TrainingSetSize"]
    trainingAnomalyIndex = [0]*HPs["TrainingSetSize"]
    for i in range(0,HPs["TrainingSetSize"]):
        trainingSet[i],trainingAnomalyIndex[i] = sampleDataSet(dimensions,
                                       trainingDataNormal,
                                       trainingDataAnomal,
                                       HPs["AnomalyPercentageTrain"],
                                       HPs["SameClassForAllDimensions"],
                                       HPs["nAnomalDimensions"],
                                       HPs["AllDimensionsAnomal"]) 
    trainingBlock = DataBlock("ECG Dataset",trainingSet,dimensions,**HPs)
    trainingBlock.setLabels(trainingAnomalyIndex)
    trainingBlock.setGeneratedFromCDS(True)

    validationSet = [0]*HPs["ValidationSetSize"]
    validationAnomalyIndex = [0]*HPs["ValidationSetSize"]
    for i in range(0,HPs["ValidationSetSize"]):
        validationSet[i],validationAnomalyIndex[i] = sampleDataSet(dimensions,
                                       validationDataNormal,
                                       validationDataAnomal,
                                       HPs["AnomalyPercentageValidation"],
                                       HPs["SameClassForAllDimensions"],
                                       HPs["nAnomalDimensions"],
                                       HPs["AllDimensionsAnomal"]) 
    
    validationBlock = DataBlock("ECG Dataset",validationSet,dimensions,**HPs)
    validationBlock.setLabels(validationAnomalyIndex)
    validationBlock.setGeneratedFromCDS(True)

    testSet = [0]*HPs["TestSetSize"]
    testAnomalyIndex = [0]*HPs["TestSetSize"]
    for i in range(0,HPs["TestSetSize"]):
        testSet[i],testAnomalyIndex[i] = sampleDataSet(dimensions,
                                       testDataNormal,
                                       testDataAnomal,
                                       HPs["AnomalyPercentageTest"],
                                       HPs["SameClassForAllDimensions"],
                                       HPs["nAnomalDimensions"],
                                       HPs["AllDimensionsAnomal"]) 

    testBlock = DataBlock("ECG Dataset",testSet,dimensions,**HPs)
    testBlock.setLabels(testAnomalyIndex)
    testBlock.setGeneratedFromCDS(True)
        
    return trainingBlock,validationBlock,testBlock
