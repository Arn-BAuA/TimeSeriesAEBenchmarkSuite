
###########################
# A Modifyer, that takes a dataset and adds uniform noise to it...

from copy import copy
from BlockAndDatablock import Datablock
import torch

def apply(modification,DataSet,newName = ""):
 
    newDSName = DataSet.Name()+newName
    dataDimensions = DataSet.Dimensions
    
    hasLabels = DataSet.hasLabels()

    if hasLabels():
        labels = copy(DataSet.Labels())    
    content = copy(DataSet.Data())
    

    for j in range(0,len(content)):
        content[j] = modification(content[j])

    newBlock = DataBlock(newDSName,content,dataDimensions)
    
    if hasLabels:
        newBlock.setLabels(labels)
    
    return newBlock

def addUniformNoise(NoiseLevelPtP,DataSet,setStd=0):
    
    if setStd ==0:
        setStd = torch.std(torch.stack(DataSet.Data()))

    def uniformNoise(dataPoint):
        dpShape = dataPoint.shape
        dataPoint += (0.5-torch.rand(dpShape))*2*(NoiseLevelPtP[i]/setStd)
        return dataPoint

    return apply(uniformNoise,DataSet," Noise "+str(NoiseLevelPtP)+" PtP")
