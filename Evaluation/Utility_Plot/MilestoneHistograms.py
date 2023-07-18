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

    Errors,minError,maxError,usableMinError,usableMaxError,maxEpoch = loadHistogramData(errorData,errorName,setName) 
    
    cmap=plt.get_cmap(CMapName,maxEpoch)
    
    #little mini view of all the errors
    insetAx = ax.inset_axes([0.6,0.7,0.35,0.25])

    for epoch in sorted(Errors):
        if not epoch == 0:
            ax.hist(Errors[epoch],
                    bins=np.linspace(usableMinError,usableMaxError,int(Errors[epoch].size/2.0)),
                    histtype=u"step",
                    density=True,
                    color=cmap(epoch),
                    #linewidth = 2
                    )
        insetAx.hist(Errors[epoch],
                bins=np.linspace(minError,maxError,int(Errors[epoch].size/2.0)),
                histtype=u"step",
                density=True,
                color=cmap(epoch),
                #linewidth = 2
                )
    #Marking Section in mini plot where huge plot
    #originated from...
    oldYLims = insetAx.get_ylim()
    
    insetAx.fill_between([usableMinError,usableMaxError],y1=oldYLims[1],y2=oldYLims[0],facecolor = "black",alpha=.2)
        
    insetAx.set_ylim(oldYLims)

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
