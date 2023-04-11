import glob
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Evaluation.Utility_Plot.General import scrapeDataFrame

###
# Script to plot the examples that where taken during the snapshots


def plotExample(rootDir,ax,ExampleName):

    data = pd.read_csv(rootDir+"/Final Model/"+ExampleName,sep='\t')
    hasTimeStamps,dataTimeStamps,hasAnomalyData,anomalyData,dataFound,dataValues = scrapeDataFrame(data,["input"])
    dataValues = dataValues[0]
    #########################
    # Plotting the Values

    if not hasTimeStamps:
        #no time stamps
        for i in range(0,dataValues.shape[-1]):
            ax.plot(dataValues[...,i],label="Dim. "+str(i))
    else:
        #Data has Timestamps
        for i in range(0,dataValues.shape[-1]):
            ax.plot(x=dataTimeStamps,y=dataValues[...,i],label="Dim. "+str(i))
    
    #########################
    # Shading values with anomalous data...
    
    if hasAnomalyData:
        oldYLims = ax.get_ylim()
        
        if hasTimeStamps:
            xVals = dataTimeStamps    
        else:
            xVals = np.arange(len(anomalyData))

        ax.fill_between(xVals,y1=oldYLims[1],y2=oldYLims[0],where=anomalyData > 0,facecolor = "black",alpha=.2)
        
        ax.set_ylim(oldYLims)

    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Values")
    ax.set_title("Data of "+ExampleName)

if __name__ == "__main__":

    if len(sys.argv)< 3:
        raise ValueError("Pleas Specify the root dir and the name of the example you want to plot")

    rootDir = sys.argv[1]
    ExampleName = sys.argv[2]

    img,ax = plt.subplots()

    plotExample(rootDir,ax,ExampleName)

    plt.show()
    plt.close()
