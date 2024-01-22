
import torch
from copy import copy
from BlockAndDatablock import DataBlock

def normalize(DataSet,newName = ""):

    newDSName = DataSet.Name()+newName
    dataDimensions = DataSet.Dimensions

    hasLabels = DataSet.hasLabels()

    if hasLabels:
        labels = copy(DataSet.Labels())

    content = torch.stack(DataSet.Data())
    std = torch.std(content)
    mean = torch.mean(content)
    
    content = torch.sub(content,mean)
    if std > 1e-12:
        content = torch.div(content,std)
    
    content = torch.unbind(content)

    newBlock = DataBlock(newDSName,content,dataDimensions)

    if hasLabels:
        newBlock.setLabels(labels)

    return newBlock

