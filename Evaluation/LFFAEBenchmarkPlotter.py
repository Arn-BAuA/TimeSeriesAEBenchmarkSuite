#Evaluation Script that goes along the LinearFeedForwardParameterVaraition script.

import sys

if len(sys.argv) == 1:
    raise ValueError("Please specify a path to a dict that was generated by the LInearFeedForwardAutoEncoderExperimentScript")

rootPath = sys.argv[1]

import os

subfolders = []

for item in os.listdir(rootPath):
    if os.path.isdir(rootPath+"/"+item):
        subfolders.append(item)

import json
import pandas as pd

data = {}

for subfolder in subfolders:
    data[subfolder]={}
    jsonFile = open(rootPath+"/"+subfolder+"/HyperParametersAndMetadata.json")
    data[subfolder]["Hyperparameters"] = json.load(jsonFile)

    modelHPs = data[subfolder]["Hyperparameters"]["Model HPs"]

    data[subfolder]["Latent Space Size"] = modelHPs["LayerSequence"][round(float(len(modelHPs["LayerSequence"]))/2.0)-1]
    data[subfolder]["Num Layers"] = len(modelHPs["LayerSequence"])+2

    data[subfolder]["Errors"] = pd.read_csv(rootPath+"/"+subfolder+"/Errors.csv",sep="\t")
    data[subfolder]["TrainingError"] = data[subfolder]["Errors"].iloc[-1]["Training Set Error"]
    data[subfolder]["ValidationError"] = data[subfolder]["Errors"].iloc[-1]["Validation Set Error"]

#Rearanging the data to be easy plottable for the first plot
ErrorPlotData = {}

for key in data:
    
    numLayers = data[key]["Num Layers"]
    
    if not numLayers in ErrorPlotData:
        ErrorPlotData[numLayers] = {}
        ErrorPlotData[numLayers]["Latent Space Size"] = [data[key]["Latent Space Size"]]
        ErrorPlotData[numLayers]["Training Set Error"] = [data[key]["TrainingError"]]
        ErrorPlotData[numLayers]["Validation Set Error"] = [data[key]["ValidationError"]]
    else:
        insertionIndex = 0

        #Insertion Sort
        inserted = False

        for i in range(0,len(ErrorPlotData[numLayers]["Latent Space Size"])):
            if ErrorPlotData[numLayers]["Latent Space Size"][i] > data[key]["Latent Space Size"]:

                ErrorPlotData[numLayers]["Latent Space Size"] = ErrorPlotData[numLayers]["Latent Space Size"][:i]+[data[key]["Latent Space Size"]]+ErrorPlotData[numLayers]["Latent Space Size"][i:]
                ErrorPlotData[numLayers]["Training Set Error"] = ErrorPlotData[numLayers]["Training Set Error"][:i]+[data[key]["TrainingError"]]+ErrorPlotData[numLayers]["Training Set Error"][i:]
                ErrorPlotData[numLayers]["Validation Set Error"] = ErrorPlotData[numLayers]["Validation Set Error"][:i]+[data[key]["ValidationError"]]+ErrorPlotData[numLayers]["Validation Set Error"][i:]
                
                inserted = True
                break
        if not inserted:
            #The new value is bigger than all values before
            ErrorPlotData[numLayers]["Latent Space Size"] = ErrorPlotData[numLayers]["Latent Space Size"]+[data[key]["Latent Space Size"]]
            ErrorPlotData[numLayers]["Training Set Error"] = ErrorPlotData[numLayers]["Training Set Error"]+[data[key]["TrainingError"]]
            ErrorPlotData[numLayers]["Validation Set Error"] = ErrorPlotData[numLayers]["Validation Set Error"]+[data[key]["ValidationError"]]



import matplotlib.pyplot as plt
import numpy as np


def plot(ErrorType,ErrorName):
    fig,ax = plt.subplots()

    for layerNum in ErrorPlotData:
        ax.plot(np.arange(len(ErrorPlotData[layerNum]["Latent Space Size"])),ErrorPlotData[layerNum][ErrorType],label="n Layers "+str(layerNum),marker = '.',linestyle = "dashed")


    ax.set_ylabel(ErrorName)
    ax.set_xlabel("Latent Space Size")

    plt.xticks(np.arange(len(ErrorPlotData[layerNum]["Latent Space Size"])),ErrorPlotData[layerNum]["Latent Space Size"])

    plt.title("Training Error VS Feed Forward Encoder Size")
    plt.legend()

    plt.show()
    plt.close()

plot("Training Set Error","Training Error")
plot("Validation Set Error","Validation Error")