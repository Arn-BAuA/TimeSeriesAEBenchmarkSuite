
from BlockAndDatablock import DataBlock
from Utility.DataSetSampler import RandomSampling
import pandas as pd
import numpy as np
import os

from Utility.MathOperations import normalize

dirname = os.path.dirname(__file__)
SMDPath = os.path.join(dirname,"../data/SMD/")

def loadData(dimensions,**hyperParameters):
    
    defaultHyperParameters = {
            "machineType":1,#Machine type of the smd dataset (number between 0 and 1)
            "machineIndex":1,#the machine index from the SMD Dataset
            "nNormalDimensions":0,#In the SMD Dataset are dimensions which don't contribute to any anomaly. these are the normal dimnsions. This algorithm selects the dimensions by anomaly. the most anomal one getts added fists, than the second and so forth. Except we demand normal dimensions by using this hyperparameter. If it is e.g. 3 we demand that the 3 least anomal dimensions are used as well.
            "ValidationSetContainsAnomalys":True,
            "ValidationSetSplit":50,# The percentage of the set where the validation set originates that is split off for the validation.
            "NormalizeValues":True,
            "TrainingSetSize":400,
            "ValidationSetSize":100,
            "TestSetSize":30,
            "SampleLength":150,# The length of the snippets that are generated.
        }

    HPs={**defaultHyperParameters,**hyperParameters}

    machineName ="machine-"+str(HPs["machineType"])+"-"+str(HPs["machineIndex"])

    #Load All The Resources...
    testData = pd.read_csv(SMDPath+"test/"+machineName+".txt",sep=',',header=None)
    trainingData = pd.read_csv(SMDPath+"train/"+machineName+".txt",sep=',',header=None)
    
    nTestLines = float(len(testData.index))

    AnomalyInDimensions = [[]]*38
    AnomalyPercentage = np.zeros(38)
    
    interpretationFile = open(SMDPath+"interpretation_label/"+machineName+".txt",'r')

    for line in interpretationFile:
        
        AnoIndices,Dimensions = line.split(':')
        AnoBegin,AnoEnd = AnoIndices.split('-')
        
        AnoBegin = int(AnoBegin)
        AnoEnd = int(AnoEnd)
        Dimensions = Dimensions.split(",")

        for dimension in Dimensions:
            d = int(dimension)
            AnomalyInDimensions[d].append({"begin":AnoBegin,"end":AnoEnd})
            AnomalyPercentage[d] += float(AnoEnd-AnoBegin)/nTestLines

    interpretationFile.close()

    #Extract which dimension is anomal on which parts
    includedDimensions = []
    
    indexPSorted = np.argsort(AnomalyPercentage) #Indices of the anomaly percentages, sorted, index of value with smallest anomaly percentage to index with largest
    
    #Collecting the normal dimensions as demanded in the HPs
    if HPs["nNormalDimensions"] < 0:
        HPs["nNormalDimensions"] = 0
    if dimensions < 0:
        dimensions = 0
    

    while len(includedDimensions) < HPs["nNormalDimensions"]:
        includedDimensions.append(indexPSorted[len(includedDimensions)])
        
    #Collecting the anomal dimensions
    while len(includedDimensions) < dimensions:
        includedDimensions.append(indexPSorted[len(indexPSorted)-1-len(includedDimensions)])

    #Create Dataframe with selected dimensoins
    testData = testData.iloc[:,includedDimensions]
    trainingData = trainingData.iloc[:,includedDimensions]
    
    if HPs["NormalizeValues"]:
        testData = normalize(testData)
        trainingData = normalize(trainingData)

    #Creating Array Marking the Anomalys on testSet
    testSetAnomalys = np.zeros(len(testData.index))
    
    for dimension in includedDimensions:
        for interval in AnomalyInDimensions[dimension]:
            testSetAnomalys[interval['begin']:interval['end']] = 1
    testData['Is Anomaly'] = testSetAnomalys
    trainingData['Is Anomaly'] = 0

    #Perform the split for the Training-, Validation- and Test-Set.
    if HPs["ValidationSetContainsAnomalys"]:
        #Take Validation Set from Testset
        testSetLen = len(testData.index)
        splitRow = int(testSetLen*(float(HPs["ValidationSetSplit"])/float(100)))
        validationData = testData.iloc[0:splitRow,:]    
        testData = testData.iloc[splitRow:testSetLen,:]
    else:
        #Take Validation Set from Trainingset
        trainingSetLen = len(trainingData.index)
        splitRow = int(trainingSetLen*(float(HPs["ValidationSetSplit"])/float(100)))
        validationData = trainingData.iloc[0:splitRow,:]
        trainingData = trainingData.iloc[splitRow:trainingSetLen,:]

     
    #Create Datasets / Splitting
    trainingData,trainingLabels = RandomSampling(trainingData,HPs["TrainingSetSize"],HPs["SampleLength"])
    trainingSet = DataBlock("SMD",trainingData,dimensions,**HPs)
    trainingSet.setLabels(trainingLabels)

    validationData,validationLabels = RandomSampling(validationData,HPs["ValidationSetSize"],HPs["SampleLength"])
    validationSet = DataBlock("SMD",validationData,dimensions,**HPs)
    validationSet.setLabels(validationLabels)
    
    testData,testLabels = RandomSampling(testData,HPs["TestSetSize"],HPs["SampleLength"])
    testSet = DataBlock("SMD",testData,dimensions,**HPs)
    testSet.setLabels(testLabels)

    return trainingSet,validationSet,testSet
