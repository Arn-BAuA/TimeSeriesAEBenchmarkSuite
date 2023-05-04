# Set Wraper for any of the datasets in the UCR Archive (https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
# The 

import pandas as pd
from random import random
import numpy as np
import torch
from BlockAndDatablock import DataBlock

def sampleDataSet(dimensions,normalData,anomalData,anomalyPercentage,allNormalTheSame,nAnomalDimensions,allDimensionsAnomal):
    
    isAnomal = random() < (float(anomalyPercentage)/100.0)
    anomalyLabel = 0
    if isAnomal:
        anomalyLabel = 1

    if allDimensionsAnomal and isAnomal:
        data = anomalData.sample(n=dimensions)
        data = data.drop(data.columns[0],axis=1)#droping the first column with the labels
        tensorData = torch.tensor(data.values.astype(np.float32))
        return torch.transpose(tensorData,0,1)
    
    normalDimensions = np.arange(0,dimensions)
    anomalDimensions = []
    
    if isAnomal:

        for i in range(0,nAnomalDimensions):
            dimensionIndex = int(len(normalDimensions)*random())
            anomalDimensions.append(normalDimensions[dimensionIndex])
            normalDimensions = np.delete(normalDimensions,dimensionIndex)
    
    if not len(anomalDimensions) <= 1:
        anomalDimensions = np.array(anomalDimensions).sort()
    

    normalSource = normalData



    if allNormalTheSame:
        firstDimension = normalData.sample()
        normalSource = normalData.loc[normalData[normalData.columns[0]] == firstDimesion.iloc[firstDimension.columns[0],0]] 
        data = pd.concat([firstDimsnion,otherDimensions])
    
    dataElements = []

    for i in range(0,dimensions):
        if i in normalDimensions:
            dataElements.append(normalSource.sample())
            continue
        if i in anomalDimensions:
            dataElements.append(anomalData.sample())

    data = pd.concat(dataElements)
    data = data.drop(data.columns[0],axis=1)#droping the first column with the labels
    tensorData = torch.tensor(data.values.astype(np.float32))
    return torch.stack([tensorData]),np.full(tensorData.size()[1],isAnomal)

        

UCRPath = "data/UCR/UCRArchive_2018/"

def loadData(dimensions,**hyperParameters):

    defaultHyperParameters = {
            "DataSet":"UMD",#Name of the dataset
            "AnomalyClass":3,#Class that is picked as anomal
            "AnomalyPercentageTrain":0,#Percentage of anomalies in training set
            "AnomalyPercentageValidation":10, #percentage of anomalies in validation set
            "AnomalyPercentageTest":10,# percentage of anomalies in test set
            "SameClassForAllDimensions":False,#when true: All dimensions are of the same class, else, they are random
            "AllDimensionsAnomal":False,#If this is true and the value is an anomaly, all the dimensions are an anomaly
            "nAnomalDimensions":1,#if it is an anomaly: How many dimensions are anomal
            "SmallestClassAsAnomaly":False, #if true, the entrie of AnomalyClass is overwritten and the smalles class is taken as anomal.
            "KeepTrainAndTestStructure":False,#if set true, the samples for training and validation are drawn from the TRAIN and TEST file in the UCR Archive. If set false, they will be mixed.
            "TrainingSetSize":400,
            "ValidationSetSize":100,
            "TestSetSize":30,
        }

    HPs={**defaultHyperParameters,**hyperParameters}
    
    trainingData = pd.read_csv(UCRPath+HPs["DataSet"]+"/"+HPs["DataSet"]+"_TRAIN.tsv",sep='\t',header=None)
    testData = pd.read_csv(UCRPath+HPs["DataSet"]+"/"+HPs["DataSet"]+"_TEST.tsv",sep='\t',header=None)
    
    #Extend to same number of Columns:

    #FillInNoneValues:

    #axis currently not implenemted in fill na...
    #trainingData.fillna(trainingData.mean(axis=1),axis=1)
    #testData.fillna(trainingData.mean(axis=1),axis=1)

    trainingData = trainingData.transpose().fillna(trainingData.mean(axis=1)).transpose()
    testData = testData.transpose().fillna(testData.mean(axis=1)).transpose()


    if not HPs["KeepTrainAndTestStructure"]:
        trainingData = pd.concat([trainingData,testData])
        testData = trainingData
    

    if HPs["SmallestClassAsAnomaly"]:
        anomalyClass = min(trainingData.iloc[1].value_counts().iloc[2])    
    else:
        anomalyClass = HPs["AnomalyClass"]
    
    HPs["ActualAnomalyClass"] = anomalyClass
    
    
    trainingAnomaly = trainingData.loc[trainingData[trainingData.columns[0]] == anomalyClass]
    trainingData = trainingData.loc[trainingData[trainingData.columns[0]] != anomalyClass]

    testAnomaly = testData.loc[testData[testData.columns[0]] == anomalyClass]
    testData = testData.loc[testData[testData.columns[0]] != anomalyClass]
     
    trainingSet = [0]*HPs["TrainingSetSize"]
    trainingAnomalyIndex = [0]*HPs["TrainingSetSize"]
    for i in range(0,HPs["TrainingSetSize"]):
        trainingSet[i],trainingAnomalyIndex[i] = sampleDataSet(dimensions,
                                       trainingData,
                                       trainingAnomaly,
                                       HPs["AnomalyPercentageTrain"],
                                       HPs["SameClassForAllDimensions"],
                                       HPs["nAnomalDimensions"],
                                       HPs["AllDimensionsAnomal"]) 
    trainingBlock = DataBlock("UCR Archive - "+HPs["DataSet"],trainingSet,dimensions,**HPs)
    trainingBlock.setLabels(trainingAnomalyIndex)

    validationSet = [0]*HPs["ValidationSetSize"]
    validationAnomalyIndex = [0]*HPs["ValidationSetSize"]
    for i in range(0,HPs["ValidationSetSize"]):
        validationSet[i],validationAnomalyIndex[i] = sampleDataSet(dimensions,
                                       trainingData,
                                       trainingAnomaly,
                                       HPs["AnomalyPercentageValidation"],
                                       HPs["SameClassForAllDimensions"],
                                       HPs["nAnomalDimensions"],
                                       HPs["AllDimensionsAnomal"]) 
    
    validationBlock = DataBlock("UCR Archive - "+HPs["DataSet"],validationSet,dimensions,**HPs)
    validationBlock.setLabels(validationAnomalyIndex)

    testSet = [0]*HPs["TestSetSize"]
    testAnomalyIndex = [0]*HPs["TestSetSize"]
    for i in range(0,HPs["TestSetSize"]):
        testSet[i],testAnomalyIndex[i] = sampleDataSet(dimensions,
                                       testData,
                                       testAnomaly,
                                       HPs["AnomalyPercentageTest"],
                                       HPs["SameClassForAllDimensions"],
                                       HPs["nAnomalDimensions"],
                                       HPs["AllDimensionsAnomal"]) 

    testBlock = DataBlock("UCR Archive - "+HPs["DataSet"],testSet,dimensions,**HPs)
    testBlock.setLabels(testAnomalyIndex)
        
    return trainingBlock,validationBlock,testBlock
