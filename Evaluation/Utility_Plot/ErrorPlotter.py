#!/bin/python

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd


def plotErrors(rootDir,ax,includerMilestones = True):
    
    errorData = pd.read_csv(rootDir+"/Errors.csv",sep="\t")
    errorData.plot(x="Epoch",y="Training Set Error",ax =ax)
    errorData.plot(x="Epoch",y="Validation Set Error",ax =ax)
    
    #marking the epochs of the goals:

    
    def plotGoals(goalFile,SetName,**plotArgs):
        labelAdded=False
        goalFile.readline()#first line with the column names
    
        for line in goalFile:
            split = line.split("\t")
            errPercentage = split[0]

            if split[1] == "NaN":
                break

            epoch = int(split[1])
        
            x=epoch
            y=errorData.iloc[epoch][SetName+" Error"]

            if not labelAdded:
                ax.scatter(x=[x],y=[y],label=SetName+" Goals",**plotArgs)
                labelAdded = True
            else:
                ax.scatter(x=[x],y=[y],**plotArgs)

            ax.text(x=x,y=y,s=errPercentage+"%")

    tsGoals = open(rootDir+"/TSGoals","r")
    vsGoals = open(rootDir+"/VSGoals","r")
    
    plotGoals(tsGoals,"Training Set",marker = "x",color="k")
    plotGoals(vsGoals,"Validation Set",marker = "o",color="k")

    tsGoals.close()
    vsGoals.close()

    ax.legend()
    ax.set_ylabel("Error")
    ax.set_title("Error on Training and Validationset during training")

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        raise ValueError("Please specify a root directory of the experiment you want to analyse.")

    rootDir = sys.argv[1]

    fig,ax = plt.subplots()

    plotErrors(rootDir,ax)

    plt.show()
    plt.close()
