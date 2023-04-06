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
            YLabels.append(column)
        if against in column:
            XLabel = column 

    for label in YLabels:
        errorData.plot(x=XLabel,y=label,ax =ax)

    ax.legend()
    ax.set_ylabel("Error")
    ax.set_xlabel(against)
    ax.set_title(errorName+" on the Sets during training")
    

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        raise ValueError("Please specify a root directory of the experiment you want to analyse.")

    rootDir = sys.argv[1]

    fig,ax = plt.subplots()

    plotErrors(rootDir,ax,"L1")

    plt.show()
    plt.close()
