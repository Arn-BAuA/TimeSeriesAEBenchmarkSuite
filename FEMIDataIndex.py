#FEMI is an acronym. It stands for Fourier Entropy mutual information.
#The Idea is to identify a Dataset with a 2D index.
# One Dimension of the index is the entropy of the fourier transformed time series of normal datapoints
# (We are asking: How much Information is there in the normal data?)
# THe Second Dimension is the Mutual Information between the fourier transformed Normal and anormal datapoints
# (We are asking: How big is the difference between normal and anomal datapoints?)

from scipy.fft import rfft,rfftfreq
import numpy as np

import matplotlib.pyplot as plt

def computeFFT(DataSet,returnPolarRepresentation=True):
    
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
            
            if returnPolarRepresentation:
                AVals = np.abs(fft[dimension,:])
                BVals = np.angle(fft[dimension,:])
            else:
                AVals = np.real(fft[dimension,:])
                BVals = np.imag(fft[dimension,:])
            

            singleDimData = np.array([AVals,BVals,fftFreq])
            singleDimData = singleDimData.T
            fftValues.append(singleDimData)

    fftValues = np.concatenate(fftValues)
    
    if returnPolarRepresentation:
        fftValues[:,0] = fftValues[:,0] * (1/maxFFTValue)
        fftValues[:,1] = fftValues[:,1] * (1/(2*np.pi))
        fftValues[:,2] = fftValues[:,2] * (1/maxFreqValue)
    else:
        fftValues[:,0] = fftValues[:,0] * (1/maxFFTValue)
        fftValues[:,1] = fftValues[:,1] * (1/maxFFTValue)
        fftValues[:,2] = fftValues[:,2] * (1/maxFreqValue)

    return fftValues

from Utility.klo.code.python.mikl import entkl as entropy

#Normal an anomal data are datablocks
def computeFEMIIndex(normalData,anomalData,polarFEMIIndex = True,noiseRegularisationMagnitude = 1e-5):

    #F =...
    normalFFTValues = computeFFT(normalData,polarFEMIIndex)
    #plt.scatter(normalFFTValues[:,2],normalFFTValues[:,0])
    #plt.show()
    #plt.close()
    anomalFFTValues = computeFFT(anomalData,polarFEMIIndex)
    #plt.scatter(anomalFFTValues[:,2],anomalFFTValues[:,0])
    #plt.show()
    #plt.close()

    #This is sort of a regularisation to ensure, that the covariace matrix has full rank...
    normalFFTValues = normalFFTValues + (np.random.rand(normalFFTValues.shape[0],normalFFTValues.shape[1])*noiseRegularisationMagnitude)
    anomalFFTValues = anomalFFTValues + (np.random.rand(anomalFFTValues.shape[0],anomalFFTValues.shape[1])*noiseRegularisationMagnitude)
    
    #EEE
    
    entropyCalculationFailed = False
    
    try:
        E_normal = entropy(normalFFTValues)
    except:
        entropyCalculationFailed = True
        E_normal = None
    
    try:
        E_anomal = entropy(anomalFFTValues)
        E_combined = entropy(np.concatenate([normalFFTValues,anomalFFTValues],axis = 1))
    except:
        entropyCalculationFailed = True

    #MI
    if not entropyCalculationFailed:
        MI = E_normal+E_anomal-E_combined
    else:
        MI = None

    return E_normal,MI

def FEMIUCRTest():
    from SetWrappers.UCRArchive import loadData as DataSet
    from SetWrappers.UCRArchive import getDatasetsInArchive
    
    dataSets = getDatasetsInArchive()
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
    

def FEMIReproducibilityTest():
    from SetWrappers.UCRArchive import loadData as DataSet
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

def FEMISinesTest():
    from DataGenerators.Sines import generateData as Sines

    print("Test on Sines")
    for AnomalyAmplitude in np.linspace(1,2,11):
        trainingSet,validationSet,testSet = Sines(dimensions=2,
                                                  AnomalousFrequency = [[0.2],[0.3]],
                                                  AnmalousAmplitudes =[[AnomalyAmplitude],[AnomalyAmplitude]],
                                                TestSetSize=200
                                                  )
     
        E,MI = computeFEMIIndex(trainingSet,testSet)
        print("Anomaly Percentage :",AnomalyAmplitude,"FEMI:",E,MI)

if __name__ == "__main__":
    #FEMISinesTest()  
    FEMIUCRTest()  
