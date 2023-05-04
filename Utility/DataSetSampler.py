

import torch
from random import random,seed
seed(1)
import pandas as pd
import numpy as np

def RandomSampling(Data,numberOfSamples,sampleWindowSize,includeTime = False,dateTimeColumn = "Date_Time"):
    DataSet = [0] * numberOfSamples
    AnomalyIndex = [0] * numberOfSamples
    sampleArea = Data.copy()
   
    if not includeTime:
        sampleArea = sampleArea.drop(columns=[dateTimeColumn])
    else:
        #conversion of datetime to timestamp for later conversion to pytorch tensor
        sampleArea[dateTimeColumn] = sampleArea[dateTimeColumn].values.astype(np.int64)
        #TODO: Noramlize timestamps to be between 0 and 1 in one window

    if len(sampleArea.index) < sampleWindowSize:
        raise Exception(f"The Samplewindow size is larger than the given range to sample")

    #Bogo (Random) Sampling...
    for i in range(0,numberOfSamples):
        
        position =int(random() * float(len(sampleArea.index)-sampleWindowSize))
        #sampling
        sequence = sampleArea.iloc[np.arange(position,position+sampleWindowSize)]
        AnomalyIndex[i] = sequence["Is Anomaly"]
        sequence = sequence.drop(columns="Is Anomaly")

        #conversion to tensor
        sampledTensor = torch.tensor(sequence.values.astype(np.float32))
        sampledTensor = torch.transpose(sampledTensor,0,1)
        DataSet[i] = torch.stack([sampledTensor])
    
    return DataSet,AnomalyIndex


