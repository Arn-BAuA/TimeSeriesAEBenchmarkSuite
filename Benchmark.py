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

dimensions = 1
sampleWindowSize = 150 # Number of samples in one Window for Training / testing

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

dataset = loadAirQualityData()
dataset.plot(x="Date_Time",y=relevantColumns)
print(dataset)
plt.show()


