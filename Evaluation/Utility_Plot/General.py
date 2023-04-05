import numpy as np

#in case that there are to many dimensions to plot, this coder
# selects the most interesting.
# it analyses the prediction of the classifier and outputs
# the dimensions with the largest, the fewest and the most
# average error.

def selectInformativeDimensions(Data,Prediction,maxDimensions):

    if Data.shape[-1] > maxDimensions:
        #determine the relevant dimensions.
        ErrorPerDimension = np.sum(np.abs((Data -Prediction)),axis = 0)
        AvgErrPerDimension = np.mean(ErrorPerDimension)
        
        numMaxErrorDims = int(float(maxDimensions)/3.0)
        numMinErrorDims = numMaxErrorDims
        numAvgErrorDims = numMaxErrorDims
        
        if maxDimensions%3 == 1:
            numMaxErrorDims+=1
        if maxDimensions%3 == 2:
            numMaxErrorDims+=1
            numAvgErrorDims+=1
        
        ErrorsSorted = np.argsort(ErrorPerDimension)
        
        relevantDims = ErrorsSorted[-numMaxErrorDims:]
        relevantDims += ErrorsSorted[:numMinErrorDims]
        relevantDims += np.argsort(np.abs(ErrorPerDimension-AvgErrPerDimension))[:numAvgErrors]
    else:
        relevantDims = np.arange(Data.shape[-1])

    return relevantDims
