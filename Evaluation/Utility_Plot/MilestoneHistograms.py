#!/bin/python
from Evaluation.Utility_Plot.General import loadHistogramData
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

CMapName = "turbo" #Cmap for the plots.

def plotMilestoneHistograms(rootDir,ax,errorName,setName):
    
    if rootDir[-1] == "/":
        rootDir = rootDir[:-1]

    errorData = pd.read_csv(rootDir+"/"+setName.replace(" ","")+"Distribution.csv",sep="\t")

    Errors,maxError,minError,maxEpoch = loadHistogramData(errorData,errorName,setName) 
    
    cmap=plt.get_cmap(CMapName,maxEpoch)

    for epoch in sorted(Errors):
        ax.hist(Errors[epoch],
                bins=np.linspace(minError,maxError,int(Errors[epoch].size/2.0)),
                histtype=u"step",
                density=True,
                color=cmap(epoch),
                #linewidth = 2
                )

    norm = mpl.colors.Normalize(vmin = 0, vmax = maxEpoch)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm = norm)
    sm.set_array([])
    plt.colorbar(sm,ticks = sorted(Errors),label="Trained Epochs",ax=ax)
    ax.set_ylabel("#Samples")
    ax.set_xlabel(errorName)
    ax.set_title("Distribution "+errorName+" on "+setName)
    

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        raise ValueError("Please specify a root directory of the experiment you want to analyse.")

    rootDir = sys.argv[1]

    fig,ax = plt.subplots()

    plotMilestoneHistograms(rootDir,ax,"L1","Test Set")

    plt.show()
    plt.close()
