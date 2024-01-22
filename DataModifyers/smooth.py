
###########################
# A Modifyer, that takes a dataset and adds uniform noise to it...

import torch
from DataModifyers.utility import apply

def rollingAverage(windowSize,DataSet,windowDim=-1):
    
    window = torch.nn.Conv1d(in_channels = DataSet.Dimension(),
            out_channels = DataSet.Dimension(),
            kernel_size = windowSize)
    window.weight.data = torch.full_like(window.weight.data,1.0/float(windowSize))

    return apply(window,DataSet," rollingAveraged WinSize: "+str(windowSize))
