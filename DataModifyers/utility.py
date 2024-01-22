
from copy import copy
from BlockAndDatablock import DataBlock
import torch

def apply(modification,DataSet,newName = ""):
 
    newDSName = DataSet.Name()+newName
    dataDimensions = DataSet.Dimensions
    
    hasLabels = DataSet.hasLabels()

    if hasLabels:
        labels = copy(DataSet.Labels())    
    content = copy(DataSet.Data())
    

    for j in range(0,len(content)):
        content[j] = modification(content[j])

    newBlock = DataBlock(newDSName,content,dataDimensions)
    
    if hasLabels:
        newBlock.setLabels(labels)
    
    return newBlock


