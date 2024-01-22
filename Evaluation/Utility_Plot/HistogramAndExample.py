
#!/bin/python
from Evaluation.Utility_Plot.General import loadHistogramData
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

def plotExampleLocation(rootDir,ax,errorName,setName,exampleIndex):
    
    if rootDir[-1] == "/":
        rootDir = rootDir[:-1]

    errorData = pd.read_csv(rootDir+"/"+setName.replace(" ","")+"Distribution.csv",sep="\t")

    Errors,minError,maxError,usableMinError,usableMaxError,maxEpoch = loadHistogramData(errorData,errorName,setName) 
    

    n,bins,patches = ax.hist(Errors[maxEpoch],
            bins=np.linspace(usableMinError,usableMaxError,int(Errors[maxEpoch].size/2.0)),
            histtype=u"step",
            #density=True,
            #linewidth = 2,
            label="Error distribution after Training",
            )

    x = Errors[maxEpoch][exampleIndex]
    ax.scatter(x,0.5,marker='x',label="Example "+str(exampleIndex))
    
    ax.legend()
    ax.set_ylabel("#Samples")
    ax.set_xlabel(errorName)
    ax.set_title("Distribution "+errorName+" on "+setName)
    

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        raise ValueError("Please specify a root directory of the experiment you want to analyse.")

    rootDir = sys.argv[1]

    fig,ax = plt.subplots()

    plotErrors(rootDir,ax,"L1","Test Set")

    plt.show()
    plt.close()
