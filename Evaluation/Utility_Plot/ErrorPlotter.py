#!/bin/python

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd


def plotErrors(rootDir,ax,errorName,against="Epoch"):
    
    errorData = pd.read_csv(rootDir+"/Errors.csv",sep="\t")

    YLabels = []
    XLabel = ""

    for column in errorData:
        if errorName in column:
            if "Delta" in column:
                pass
                #TODO: Routine here to plot deviation
            else:
                YLabels.append(column)
        if against in column:
            XLabel = column 

    for label in YLabels:
        errorData.plot(x=XLabel,y=label,ax =ax)

    ax.legend()
    ax.set_ylabel("Error")
    ax.set_xlabel(against)
    ax.set_title(errorName+" on the Sets during training")

def plotErrorsAgainstExample(rootDir,ax,errorName,setName,exampleIndex):
 
    errorData = pd.read_csv(rootDir+"/Errors.csv",sep="\t")

    YLabel = ""

    for column in errorData:
        if errorName in column and setName in column:
            if not "Delta" in column:
                YLabel = column
    errorData.plot(y=YLabel,ax =ax,label = "Average Error")

    distributionData = pd.read_csv(rootDir+"/"+setName+"SetDistribution.csv",sep="\t")
    
    epochs = []
    for column in distributionData:
        if  not errorName in column:
            distributionData.drop(columns = column)
        else:
            epochs.append(int(column.split(" ")[1]))
    
    ax.scatter(x=epochs,y=distributionData.iloc[exampleIndex].to_numpy()[1:],label="Error for Example Datapoint")

    ax.legend()
    ax.set_ylabel(errorName)
    ax.set_xlabel("Epochs")
    ax.set_title(errorName+" on the Sets during training")




if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        raise ValueError("Please specify a root directory of the experiment you want to analyse.")

    rootDir = sys.argv[1]

    fig,ax = plt.subplots()

    plotErrors(rootDir,ax,"L1")

    plt.show()
    plt.close()
