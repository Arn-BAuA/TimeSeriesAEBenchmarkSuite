#!/bin/bash

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
import os
#####
# THis Script plots the 

CMapName = "turbo" #Cmap for the plots.

def plotMilestones(rootDir,ax,ExampleName):
    
    rootDir = rootDir[:-1]

    milestoneEpochs = []
    milestoneFiles = []
    
    trueDataTimestamps = []
    trueData = []

    for item in os.listdir(rootDir+"/Milestones"):
        if os.path.isdir(rootDir+"/Milestones/"+item):
            split = item.split(" ")
            
            if split[0] == "Milestone": #Dir is indeed a dir containing milestones
                milestoneEpochs.append(int(split[2]))
                milestoneFiles.append(rootDir+"/Milestones/"+item+"/"+ExampleName)
        
    #Getting the Last Epoch:
    errFile = open(rootDir+"/Errors.csv","r")
    last_line = errFile.readlines()[-1]
    split = last_line.split("\t")

    milestoneEpochs.append(int(split[0]))
    milestoneFiles.append(rootDir+"/Final Model/"+ExampleName)
        
    AEGeneratedData = [0]*len(milestoneEpochs)
    
    for i in range(0,len(milestoneEpochs)):
        
        data = pd.read_csv(milestoneFiles[i],sep='\t')
        
        for column in data:
            if not "output" in column:
                data=data.drop(columns=column)

        AEGeneratedData[i] = data.to_numpy()

    #Loading true data...
    data = pd.read_csv(milestoneFiles[0],sep='\t')
    
    for column in data:
        if "time" in column:
            trueDataTimestamps = data[column].to_numpy()
            data=data.drop(columns=column)
            continue
        if not "input" in column:
            data=data.drop(columns=column)
    trueData = data.to_numpy()

    ###########################################
    #       Data Loaded, now plot             #
    ###########################################
    maxEpoch = milestoneEpochs[-1]
    
    cmap=plt.get_cmap(CMapName,maxEpoch)

    def plotDataFrame(timeStamps,data,**plotArgs):
        if len(timeStamps) == 0:
            #no time stamps.
            for i in range(0,data.shape[-1]):
                ax.plot(data[...,i],**plotArgs)
        else:
            #with time stamps
            for i in range(0,data.shape[-1]):
                ax.plot(x=timeStamps,y=data[...,i],**plotArgs)

    for i in range(0,len(AEGeneratedData)):
        plotDataFrame(trueDataTimestamps,AEGeneratedData[i],color=cmap(milestoneEpochs[i]))

    plotDataFrame(trueDataTimestamps,trueData,color="k",linestyle="dashed",linewidth = 2,label="Original Data")
    
    ax.legend()

    #plotting color bar
    norm = mpl.colors.Normalize(vmin = 0, vmax = maxEpoch)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm = norm)
    sm.set_array([])
    plt.colorbar(sm,ticks = milestoneEpochs,label="Trained Epochs")

    ax.set_xlabel("Time")
    ax.set_ylabel("Data")

    ax.set_title("Output of the AE VS original Data")

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 3:
        raise ValueError("Please specify the path to the directory with the rundata you want to plot milestones from and the name of the desired example file for each milestone")


    rootDir = sys.argv[1]
    ExampleName = sys.argv[2]

    img,ax = plt.subplots()

    plotMilestones(rootDir,ax,ExampleName)

    plt.show()
    plt.close()
