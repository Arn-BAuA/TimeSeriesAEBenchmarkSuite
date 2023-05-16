#FEMI is an acronym. It stands for Fourier Entropy mutual information.
#The Idea is to identify a Dataset with a 2D index.
# One Dimension of the index is the entropy of the fourier transformed time series of normal datapoints
# (We are asking: How much Information is there in the normal data?)
# THe Second Dimension is the Mutual Information between the fourier transformed Normal and anormal datapoints
# (We are asking: How big is the difference between normal and anomal datapoints?)

from scipy.fft import rfft,rfftfreq
import numpy as np

def computeFFT(DataSet):
    
    fftValues = [] 
    
    maxFFTValue = 0
    maxFreqValue = 0

    for dataPoint in DataSet.Data():

        DPAsNumpy = dataPoint[0,:,:].to("cpu").detach().numpy()
        
        #Multiplying with Hanning Window:
        window = np.hanning(DPAsNumpy.shape[-1])
        DPAsNumpy = np.multiply(DPAsNumpy,window)

        #Copmute FFT
        fft = rfft(DPAsNumpy)
        fftFreq = rfftfreq(DPAsNumpy.shape[-1])
        
        #Gather Maxima for normalisation
        maximumVal = np.max(np.absolute(fft))
        if maximumVal > maxFFTValue:
            maxFFTValue = maximumVal

        maxFreq = max(fftFreq)
        if maxFreq > maxFreqValue:
            maxFreqValue = maxFreq
        
        #Reshaping:
        for dimension in range(0,DataSet.Dimension()):
            
            ReVals = np.real(fft[dimension,:])
            ImVals = np.imag(fft[dimension,:])
            

            singleDimData = np.array([ReVals,ImVals,fftFreq])
            singleDimData = singleDimData.T
            fftValues.append(singleDimData)

    fftValues = np.concatenate(fftValues)
    
    fftValues[:,0] = fftValues[:,0] * (1/maxFFTValue)
    fftValues[:,1] = fftValues[:,1] * (1/maxFreqValue)

    return fftValues

from Utility.klo.code.python.mikl import entkl as entropy

#Normal an anomal data are datablocks
def computeFEMIIndex(normalData,anomalData):

    #F =...
    normalFFTValues = computeFFT(normalData)
    anomalFFTValues = computeFFT(anomalData)

    normalFFTValues = normalFFTValues + (np.random.rand(normalFFTValues.shape[0],normalFFTValues.shape[1])*1e-5)
    anomalFFTValues = anomalFFTValues + (np.random.rand(anomalFFTValues.shape[0],anomalFFTValues.shape[1])*1e-5)
    
    #print(normalFFTValues.shape)
    #halt
    E_normal = entropy(normalFFTValues)


#    if not normalFFTValues.shape[0] == anomalFFTValues.shape[0]:
#        commonLength = min([normalFFTValues.shape[0],anomalFFTValues.shape[0]])
#        normalFFTValues = normalFFTValues[:commonLength]
#        anomalFFTValues = anomalFFTValues[:commonLength]

#   MI = mutualInformation(normalFFTValues,anomalFFTValues)

    E_anomal = entropy(anomalFFTValues)
    E_combined = entropy(np.concatenate([normalFFTValues,anomalFFTValues],axis = 1))


    MI = E_normal+E_anomal-E_combined
    
    return E_normal,MI

if __name__ == "__main__":
    from SetWrappers.UCRArchive import loadData as DataSet
    from SetWrappers.UCRArchive import getDatasetsInArchive
    
    dataSets = getDatasetsInArchive()

    print("Reproducibility Test:")
    for setSize in [10,100,1000]:
        valuesE = []
        valuesMI = []
        for i in range(0,10):
            trainingSet,validationSet,testSet = DataSet(
                                                            dimensions=1,
                                                            TrainingSetSize = setSize,
                                                            ValidationSetSize = setSize,
                                                            TestSetSize = 0)
    
            E,MI = computeFEMIIndex(trainingSet,validationSet)
            valuesE.append(E)
            valuesMI.append(MI)
        
        print("For ",setSize," samples: E:",np.mean(valuesE)," +/- ",np.std(valuesE),"MI: ",np.mean(valuesMI)," +/- ",np.std(valuesMI))

    print("Test: on UCR")

    for dsName in dataSets:
        anomalyPercentage = [0,30,50,70,100]
        print(dsName+":")
        
        for a in anomalyPercentage:
            trainingSet,validationSet,testSet = DataSet(
                                                        DataSet = dsName,
                                                        dimensions=1,
                                                        TrainingSetSize = 100,
                                                        ValidationSetSize = 100,
                                                        TestSetSize = 0,
                                                        anomalyPercentageTest = a)
    
            E,MI = computeFEMIIndex(trainingSet,validationSet)
            print("Anomaly Percentage :",a,"FEMI:",E,MI)

