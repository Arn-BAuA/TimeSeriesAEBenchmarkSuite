
###########################
# A Modifyer, that takes a dataset and adds uniform noise to it...

import torch
from DataModifyers.utility import apply

def addUniformNoise(NoiseLevelPtP,DataSet,setStd=0):
    
    if setStd ==0:
        setStd = torch.std(torch.stack(DataSet.Data()))

    def uniformNoise(dataPoint):
        dpShape = dataPoint.shape
        dataPoint += (0.5-torch.rand(dpShape))*2*(NoiseLevelPtP/setStd)
        return dataPoint

    return apply(uniformNoise,DataSet," Noise "+str(NoiseLevelPtP)+" PtP")
