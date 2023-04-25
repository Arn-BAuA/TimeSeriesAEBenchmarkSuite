# Set Wraper for any of the datasets in the UCR Archive (https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
# The 

import pandas as pd
from random import random
import numpy as np
import torch
from BlockAndDatablock import DataBlock

def sampleDataSet(dimensions,normalData,anomalData,anomalyPercentage,allNormalTheSame,nAnomalDimensions,allDimensionsAnomal):
    
    isAnomal = random() < (float(anomalyPercentage)/100.0)

    if allDimensionsAnomal and isAnomal:
        data = anomalData.sample(n=dimensions)
        data = data.drop(data.columns[1],axis=1)#droping the first column with the labels
        tensorData = torch.tensor(data.values.astype(np.float32))
        return torch.transpose(tensorData,0,1)
        
    if allNormalTheSame:
        firstDimension = normalData.sample()
        otherDimensions = normalData.loc[normalData[normalData.columns[0]] == firstDimesion.iloc[firstDimension.columns[0],0]].sample(n=dimensions-1) 
        data = pd.concat([firstDimsnion,otherDimensions])
    else:
        data = normalData.sample(n=dimensions)

    if isAnomal:
        normalDimensions = np.arange(0,dimensions)

        for i in range(0,nAnomalDimensions):
            dimensionIndex = int(len(normalDimensions)*random())
            dimension = normalDimensions[dimensionIndex]
            normalDimensions = np.delete(normalDimensions,dimensionIndex)

            sample = anomalData.sample().values
            print(data)
            print(sample)
            print(data.loc[dimension])

            data.loc[dimension] = sample

    
    data = data.drop(data.columns[1],axis=1)#droping the first column with the labels
    tensorData = torch.tensor(data.values.astype(np.float32))
    return torch.transpose(tensorData,0,1)

        

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
    
    trainingData = pd.read_csv(UCRPath+HPs["DataSet"]+"/"+HPs["DataSet"]+"_TRAIN.tsv",sep='\t')
    testData = pd.read_csv(UCRPath+HPs["DataSet"]+"/"+HPs["DataSet"]+"_TEST.tsv",sep='\t')
    
    if not HPs["KeepTrainAndTestStructure"]:
        trainingData = pd.concat([trainingData,testData])
        testData = trainingData
    


    if HPs["SmallestClassAsAnomaly"]:
        anomalyClass = min(trainingData.iloc[1].value_counts().iloc[2])    
    else:
        anomalyClass = HPs["AnomalyClass"]

    
    
    trainingAnomaly = trainingData.loc[trainingData[trainingData.columns[0]] == anomalyClass]
    trainingData = trainingData.loc[trainingData[trainingData.columns[0]] != anomalyClass]

    testAnomaly = testData.loc[testData[testData.columns[0]] == anomalyClass]
    testData = testData.loc[testData[testData.columns[0]] != anomalyClass]
    
    trainingSet = [0]*HPs["TrainingSetSize"]
    for i in range(0,HPs["TrainingSetSize"]):
        trainingSet[i] = sampleDataSet(dimensions,
                                       trainingData,
                                       trainingAnomaly,
                                       HPs["AnomalyPercentageTrain"],
                                       HPs["SameClassForAllDimensions"],
                                       HPs["nAnomalDimensions"],
                                       HPs["AllDimensionsAnomal"]) 
    
    trainingBlock = DataBlock("UCR Archive - "+HPs["DataSet"],trainingSet,dimensions,**HPs)

    validationSet = [0]*HPs["ValidationSetSize"]
    for i in range(0,HPs["ValidationSetSize"]):
        validationSet[i] = sampleDataSet(dimensions,
                                       trainingData,
                                       trainingAnomaly,
                                       HPs["AnomalyPercentageValidation"],
                                       HPs["SameClassForAllDimensions"],
                                       HPs["nAnomalDimensions"],
                                       HPs["AllDimensionsAnomal"]) 
    
    validationBlock = DataBlock("UCR Archive - "+HPs["DataSet"],validationSet,dimensions,**HPs)

    testSet = [0]*HPs["TestSetSize"]
    for i in range(0,HPs["TestSetSize"]):
        testSet[i] = sampleDataSet(dimensions,
                                       testData,
                                       testAnomaly,
                                       HPs["AnomalyPercentageTest"],
                                       HPs["SameClassForAllDimensions"],
                                       HPs["nAnomalDimensions"],
                                       HPs["AllDimensionsAnomal"]) 
    
    testBlock = DataBlock("UCR Archive - "+HPs["DataSet"],testSet,dimensions,**HPs)

    return trainingBlock,validationBlock,testBlock
