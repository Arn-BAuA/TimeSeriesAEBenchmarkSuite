
import pandas as pd
from datetime import date,time,datetime
from random import random, seed
import numpy as np
import torch


PathToAirqualityData = "data/AirQualityUCI.xlsx"

#These Columns contain approximated values for some Metals
# in the air. They look nice (smooth with bounded rate of 
#change and absolute values) The data is not clean.
# values are usually between 600 and 2000, sometimes they
#are constant 0 when the station is down. POint are 
# equidistant in time.
Columns = [
        "PT08.S1(CO)",
        "PT08.S2(NMHC)",
        "PT08.S3(NOx)",
        "PT08.S4(NO2)",
        "PT08.S5(O3)",
        ]

def loadData(savePath,**hyperParameters):
    
    defaultHyperParameters = {
            "Dimensions" = 1,
            "sampleWindowSize" = 150,
            "includeTimeStamps" =False,
            "TrainingSetSize" = 1000,
            "BeginDateTrainingData" = datetime(2004,4,1),
            "EndDateTrainingData" = datetime(2005,1,1),
            "ValidationSetSize" = 100,
            "BeginDateValidationData" = datetime(2005,1,1),
            "EndDateValidationData" = datetime(2005,3,1),
            "TestSetSize" = 30,
            "BeginDateTestData" = datetime(2005,3,1),
            "EndDateTestData" = datetime(2005,4,1),
            }
    
    HPs = {**defaultHyperParameters,**hyperParameters}
    
    #################################
    # Loading and Cleaning the Data #
    #################################


    relevantColumns = Columns[0:HPs["Dimensions"]]

    #loading and preparing air quality data
    allData = pd.read_excel(PathToAirqualityData,parse_dates=[["Date","Time"]])
    allData = allData[["Date_Time"]+relevantColumns]
    #normalisation for better numeric stability
    allData[relevantColumns] = (allData[relevantColumns]-allData[relevantColumns].mean())/allData[relevantColumns].std()
    

    #################################
    # Sampling the DataSets         #
    #################################


    seed(1)

    def SampleDataSet(beginDate,endDate,numberOfSamples):
    
        DataSet = [0] * numberOfSamples
        sampleArea = allData.loc[(allData["Date_Time"]>beginDate) & (allData["Date_Time"]<= endDate)]
   
        if not HPs["includeTimeStamps"]:
            sampleArea = sampleArea.drop(columns=["Date_Time"])
        else:
            #conversion of datetime to timestamp for later conversion to pytorch tensor
            sampleArea["Date_Time"] = sampleArea.Date_Time.values.astype(np.int64)

        if len(sampleArea.index) < HPs["sampleWindowSize"]:
            raise Exception(f"The Samplewindow size is larger than the given range to sample from ({sampleWindowSize}/{len(sampleArea.index)})")

        #Bogo (Random) Sampling...
        for i in range(0,numberOfSamples):
        
            position =int(random() * float(len(sampleArea.index)-HPs["sampleWindowSize"]))
            #sampling
            sequence = sampleArea.iloc[np.arange(position,position+HPs["sampleWindowSize"])]
            #conversion to tensor
            DataSet[i] = torch.tensor(sequence.values.astype(np.float32))
            #DataSet[i] = torch.transpose(DataSet[i],0,1)
    
        return DataSet

    trainingSet = SampleDataSet(HPs["BeginDateTrainingData"],HPs["EndDateTrainingData"],HPs["TrainingSetSize"])
    validationSet = SampleDataSet(HPs["BeginDateValidationData"],HPs["EndDateValidationData"],HPs["ValidationSetSize"])
    testSet = SampleDataSet(HPs["BeginDateTestData"],HPs["EndDateTestData"],HPs["TestSetSize"])

    return trainingSet,validationSet,testSet
