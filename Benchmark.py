#!/bin/python

import pandas as pd
from datetime import date,time,datetime
import matplotlib.pyplot as plt

PathToAirqualityData = "data/AirQualityUCI.xlsx"
Columns = [
        "PT08.S1(CO)",
        "PT08.S2(NMHC)",
        "PT08.S3(NOx)",
        "PT08.S4(NO2)",
        "PT08.S5(O3)",
        ]

dimensions = 2
sampleWindowSize = 150 # Number of samples in one Window for Training / testing
useTimestapmsAsInput = True

#################################
# Data Loading                  #
#################################

relevantColumns = Columns[0:dimensions]

#loading and preparing air quality data
def loadAirQualityData():
    dataset = pd.read_excel(PathToAirqualityData,parse_dates=[["Date","Time"]])
    print(relevantColumns)
    dataset = dataset[["Date_Time"]+relevantColumns]
    return dataset

allData = loadAirQualityData()

#allData.plot(x="Date_Time",y=relevantColumns)
#print(allData)
#plt.show()

#################################
# Sampling the DataSets         #
#################################

from random import random, seed
import numpy as np
import torch

seed(1)

####################
# Take snippets at random locations from the dataset
#
#
def SampleDataSet(beginDate,endDate,numberOfSamples):
    
    DataSet = [0] * numberOfSamples
    sampleArea = allData.loc[(allData["Date_Time"]>beginDate) & (allData["Date_Time"]<= endDate)]
    
    if len(sampleArea.index) < sampleWindowSize:
        raise Exception(f"The Samplewindow size is larger than the given range to sample from ({sampleWindowSize}/{len(sampleArea.index)})")

    #Bogo (Random) Sampling...
    for i in range(0,numberOfSamples):
        
        position =int(random() * float(len(sampleArea.index)-sampleWindowSize))
        
        #sampling
        sequence = sampleArea.iloc[np.arange(position,position+sampleWindowSize)]
        #conversion to tensor
        sequence = sequence.astype(np.float32).to_numpy().tolist()
        DataSet[i] = torch.tensor(sequence)
    
    return DataSet

ts = SampleDataSet(datetime(2004,4,1),datetime(2005,1,1),10)

print(ts)
print(ts[0])
ts[0].plot(x="Date_Time",y=relevantColumns)
plt.show()



