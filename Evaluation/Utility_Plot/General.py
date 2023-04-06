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


##################
# Method for scraping thee relevant data from the loaded
# dataframes
def scrapeDataFrame(data,relevantCols,ignoreTime = False,ignoreLabels = False):

    relevantData = [0]*len(relevantCols)
    DataFound = [False]*len(relevantCols)
    if not ignoreTime:
        hasTimeStamps = False
        dataTimeStamps = []
    if not ignoreLabels:
        hasLabels = False
        labels = []

    for column in data:
        if not ignoreTime and "time" in column:
            dataTimeStamps = data[column].to_numpy()
            hasTimeStamps = True
            continue
        if not ignoreLabels and "Anomaly" in column or "anomaly" in column:
            labels = data[column].to_numpy()
            hasLabels = True
            continue
        for i,identifier in enumerate(relevantCols):
            if identifier in column:
                if DataFound[i]:
                    relevantData[i][column] = data[column]
                else:
                    relevantData[i] = data[column].to_frame()
                    DataFound[i] = True
                continue
    
    for i in range(0,len(relevantData)):
        relevantData[i] = relevantData[i].to_numpy()
    
    toReturn = [DataFound,relevantData]
    if not ignoreLabels:
        toReturn = [hasLabels,labels]+toReturn
    if not ignoreTime:
        toReturn = [hasTimeStamps,dataTimeStamps] + toReturn

    return (*toReturn,)
