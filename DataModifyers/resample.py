
##############################
# Interpolates the time siereis in a datablock to be a different number of time poinst. Basically a wrapper for torch.nn.functional.interpolate

import torch
import numpy as np
from DataModifyers.utility import applyToDataAndLabels

def resample(newNrOfTimePoints,DataSet,mode="linear"):
    
    def interpolateData(DataPoint):
        return torch.nn.functional.interpolate(DataPoint,size=newNrOfTimePoints,mode=mode)
    

    # I am sure there is a better way, using a numpy build in function... What ever.
    def interpolateLables(LabelArray):
        
        LabeledPositions = np.linspace(0,1,len(LabelArray))
        NewPositions = np.linspace(0,1,newNrOfTimePoints)
        
        newLabels = np.zeros(newNrOfTimePoints)

        oldPositionIndex = 0
        
        for i in range(0,len(NewPositions)):
            
            while (not oldPositionIndex == len(LabeledPositions)-2) and (NewPositions[i] > LabeledPositions[oldPositionIndex]):
                oldPositionIndex += 1
            
            distanceLeft = NewPositions[i]-LabeledPositions[oldPositionIndex]
            distanceRight = LabeledPositions[oldPositionIndex+1]-NewPositions[i+1]
            
            if distanceRight >= distanceLeft:
                newLabels[i] = LabelArray[oldPositionIndex]
            else:
                newLabels[i] = LabelArray[oldPositionIndex+1]

        return newLabels

    return applyToDataAndLabels(interpolateData,interpolateLables,DataSet)
