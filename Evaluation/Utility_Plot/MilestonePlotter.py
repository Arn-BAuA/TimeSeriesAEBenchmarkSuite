#!/bin/bash

from Evaluation.Utility_Plot.General import selectInformativeDimensions,scrapeDataFrame
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
import os
#####
# THis Script plots the 

CMapName = "turbo" #Cmap for the plots.

def plotMilestones(rootDir,ax,ExampleName,maxDimensions = 4):
    
    rootDir = rootDir[:-1]

    milestoneEpochs = []
    milestoneFiles = []
    
    trueDataTimestamps = []
    trueData = []

    for item in os.listdir(rootDir+"/Milestones"):
        if os.path.isdir(rootDir+"/Milestones/"+item):
            split = item.split(" ")
            
            if split[0] == "Milestone":
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

        outputFound,OutputData = scrapeDataFrame(data,["output"],ignoreTime = True,ignoreLabels=True)
        AEGeneratedData[i] = OutputData[0]
    

    #Loading true data...
    data = pd.read_csv(milestoneFiles[0],sep='\t')
   
    hasTimeStamps,trueDataTimestamps,hasLabels,labels,hasTrueData,trueData = scrapeDataFrame(data,["input"]) 
    trueData = trueData[0]

    ###########################################
    #       Data Loaded, now plot             #
    ###########################################
    maxEpoch = milestoneEpochs[-1]
    
    cmap=plt.get_cmap(CMapName,maxEpoch)
    
    relevantDims = selectInformativeDimensions(trueData,AEGeneratedData[-1],maxDimensions)

    def plotDataFrame(timeStamps,data,**plotArgs):
        firstPlotDone = False
        if len(timeStamps) == 0:
            #no time stamps.
            for dim in relevantDims:
                ax.plot(data[...,dim],**plotArgs)
                if not firstPlotDone:
                    firstPlotDone = True
                    if "label" in plotArgs:
                        del plotArgs["label"]
        else:
            #with time stamps
            for dim in relevantDims:
                ax.plot(x=timeStamps,y=data[...,dim],**plotArgs)
                if not firstPlotDone:
                    firstPlotDone = True
                    if "label" in plotArgs:
                        del plotArgs["label"]


    for i in range(0,len(AEGeneratedData)):
        plotDataFrame(trueDataTimestamps,AEGeneratedData[i],color=cmap(milestoneEpochs[i]))

    plotDataFrame(trueDataTimestamps,trueData,color="k",linestyle="dashed",linewidth = 2,label="Original Data")
    
    
    if hasLabels:
        oldYLims = ax.get_ylim()
        
        if hasTimeStamps:
            xVals = trueDataTimestamps 
        else:
            xVals = np.arange(len(labels))

        ax.fill_between(xVals,y1=oldYLims[1],y2=oldYLims[0],where=labels > 0,facecolor = "black",alpha=.2)
        
        ax.set_ylim(oldYLims)



    ax.legend()

    #plotting color bar
    norm = mpl.colors.Normalize(vmin = 0, vmax = maxEpoch)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm = norm)
    sm.set_array([])
    plt.colorbar(sm,ticks = milestoneEpochs,ax=ax,label="Trained Epochs")

    ax.set_xlabel("Time")
    ax.set_ylabel("Data")

    ax.set_title("Output of the Model VS Original Data")

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
