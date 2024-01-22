

import torch
from random import random,seed
seed(1)
import pandas as pd
import numpy as np

def RandomSampling(Data,numberOfSamples,sampleWindowSize,includeTime = False,dateTimeColumn = "Date_Time"):
    DataSet = [0] * numberOfSamples
    AnomalyIndex = [0] * numberOfSamples
    sampleArea = Data.copy()
   
    if not includeTime:
        if dateTimeColumn in sampleArea.columns:
            sampleArea = sampleArea.drop(columns=[dateTimeColumn])
    else:
        #conversion of datetime to timestamp for later conversion to pytorch tensor
        sampleArea[dateTimeColumn] = sampleArea[dateTimeColumn].values.astype(np.int64)
        #TODO: Noramlize timestamps to be between 0 and 1 in one window

    if len(sampleArea.index) < sampleWindowSize:
        raise Exception(f"The Samplewindow size is larger than the given range to sample")

    #Bogo (Random) Sampling...
    for i in range(0,numberOfSamples):
        
        position =int(random() * float(len(sampleArea.index)-sampleWindowSize))
        #sampling
        sequence = sampleArea.iloc[np.arange(position,position+sampleWindowSize)]
        AnomalyIndex[i] = sequence["Is Anomaly"]
        sequence = sequence.drop(columns="Is Anomaly")

        #conversion to tensor
        sampledTensor = torch.tensor(sequence.values.astype(np.float32))
        sampledTensor = torch.transpose(sampledTensor,0,1)
        DataSet[i] = torch.stack([sampledTensor])
    
    return DataSet,AnomalyIndex

def fromClassification2AnoDetect(dimensions,normalData,anomalData,anomalyPercentage,allNormalTheSame,nAnomalDimensions,allDimensionsAnomal):
    
    
    isAnomal = random() < (float(anomalyPercentage)/100.0)
    anomalyLabel = 0
    if isAnomal:
        anomalyLabel = 1

    if allDimensionsAnomal and isAnomal:
        data = anomalData.sample(n=dimensions)
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
    tensorData = torch.tensor(data.values.astype(np.float32))
    return torch.stack([tensorData]),np.full(tensorData.size()[1],isAnomal)


def splitDataframeAlongRows(percentages,df,randomlySampleBeforeSplit = True):

    nRows = len(df.index)

    if randomlySampleBeforeSplit:
        df = df.sample(frac=1)

    #the indices at which we "cut" the dataframe
    cuttedDFs = [0]*len(percentages)
    currentCutPosition = int(np.round(percentages[0]*nRows))
    cuttedDFs[0] = df.iloc[0:currentCutPosition,:]

    for i in range(1,len(percentages)):

        if i == len(percentages)-1:
            #to prevent rounding errors
            newCutPosition = nRows
        else:
            newCutPosition = currentCutPosition + int(np.round(percentages[i])*nRows)


        cuttedDFs[i] = df.iloc[currentCutPosition:newCutPosition,:]

        currentCutPosition = newCutPosition


    return (*cuttedDFs,)

#returns a copy of df, containing only the rows of df, where the value of the label column is in labels
def selectByLabel(df,labels,labelColumn,deleteLabelColumn = False):

    dfComponentsByLabel = [0]*len(labels)

    for i,label in enumerate(labels):
        dfComponentsByLabel[i] = df[df[labelColumn] == label]

    filteredDf = pd.concat(dfComponentsByLabel)

    if deleteLabelColumn:
        filteredDf = filteredDf.drop(labelColumn,axis = 1)

    return filteredDf
