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


def loadHistogramData(errorData,errorName,setName):

    Errors = {}
    
    maxError = -1e9
    minError = 1e9
    maxEpoch = 0

    for column in errorData:
        if errorName in column:
            #Milestone Extrahieren und als Key...
            Epoch = int(column.split(" ")[1])
            Errors[Epoch] = errorData[column].to_numpy()
            
            if Epoch != 0:
                localMax = np.max(Errors[Epoch])
                if localMax > maxError:
                    maxError = localMax
                localMin = np.min(Errors[Epoch])
                if localMin < minError:
                    minError = localMin
            if Epoch > maxEpoch:
                maxEpoch = Epoch
    
    Errors[0] = Errors[0][np.where((Errors[0] >= minError) & (Errors[0]<= maxError))]
    
    if len(Errors[0]) == 0:
        del Errors[0]

    return Errors,maxError,minError,maxEpoch


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
