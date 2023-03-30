#!/bin/python

################################################
# THis is a Datagenerator, that adds a bunch of sine waves with noise and drift.


from BlockAndDatablock import Datablock
import numpy as np
from numpy.random import random
import pandas as pd
from Utility.DataSetSampler import RandomSampling

def getSingleParameterVariation(value,span,changeRate,MaxSystemVelocity,duration,time):
    nChanges = duration*changeRate

    timePoints = np.random([nChanges])*duration
    timePoints = timePoints.sort()
    timePoints[0]=0
    timePoints[-1] = duration

    timeIntervals = timePoints[1:] - timePoints[:-1]
    possibleMaxChange = timeIntervals*MaxSystemVelocity
    
    parameterValues = np.zeros(nChanges)

    #Random Walk:
    randomWalkValues[0] = (random()*2*span)-span #initial Parameter

    
    change = np.multiply(2*possibleMaxChange,random([nChanges-1]))-possibleMaxChange

    for i in range(1,len(randomWalkValues)):
        
        if change[i-1] > 2*span:
            newParameter = (2*span*random())-span
        else:
            newParameter = randomWalkValues[i-1] + change[i-1]

            if newParameter < -span:
                newParameter = -span -(newParameter+span)
            if newParameter > span:
                mewParameter = span - (newParameter-span)

        randomWalkValues[i] = newParameter

    #Getting parameter Varaition by linear interpolation between the points of the walk.
    parameterValues = np.zeros(len(time))

    for i in range(0,len(time)):
        t1Index = np.argwhere(timePoints <= time[i]).max() #Since they are sorted, returns the index of the largest t value smaller time[i]
        t2Index = np.argwhere(timePoints >= time[i]).max()

        t1 = timePoints[t1Index]
        v1 = randomWalkValues[t1Index]
        t2 = timePoints[t2Index]
        v2 = randomWalkValues[t2Index]

        #LinearInterpolation:
        parameterValues[i] = v1*((t2-time[i])/(t2-t1)) + v2 * ((time[i]-t1)/(t2-t1))

    return parameterValues+value
        

def generate1DSines(**HPs,BlendArray,time):
    
    sumOfSines = np.zeros(len(time))

    for i in range(0,len(HPs["Amplitudes"])):
        
        #Amplitude
        NormalAmplitudeValues = getSingleParameterVariation(HPs["Amplitudes"][i],
                                                            HPs["AmplitudeSpan"][i],
                                                            HPs["SystemChangeRate"],
                                                            HPs["MaxSystemVelocity"],
                                                            HPs["Duration"],
                                                            time)

        AnomalAmplitudeValues = getSingleParameterVariation(HPs["AnomalousAmplitudes"][i],
                                                            HPs["AnomalousAmplitudeSpan"][i],
                                                            HPs["SystemChangeRate"],
                                                            HPs["MaxSystemVelocity"],
                                                            HPs["Duration"],
                                                            time)

        Amplitude = np.multiply(NormalAmplitudeValues,np.ones(len(BlendArray))-(BlendArray*HPs["AnomalyMagnitudeInAmplitude"])) + np.multiply(AnomalAmplitudeValues,BlendArray*HPs["AnomalyMagnitudeInAmplitude"])

        #Frequency
        NormalFrquencyValues = getSingleParameterVariation(HPs["Frequency"][i],
                                                            HPs["FrequencySpan"][i],
                                                            HPs["SystemChangeRate"],
                                                            HPs["MaxSystemVelocity"],
                                                            HPs["Duration"],
                                                            time)

        AnomalFrequencyValues = getSingleParameterVariation(HPs["AnomalousFrequency"][i],
                                                            HPs["AnomalousFrequencySpan"][i],
                                                            HPs["SystemChangeRate"],
                                                            HPs["MaxSystemVelocity"],
                                                            HPs["Duration"],
                                                            time)

        Frequency = np.multiply(NormalFrequencyValues,np.ones(len(BlendArray))-(BlendArray*HPs["AnomalyMagnitudeInFrequency"])) + np.multiply(AnomalFrequencyValues,BlendArray*HPs["AnomalyMagnitudeInFrequency"])


        #Offset
        NormalOffsetValues = getSingleParameterVariation(HPs["Offset"][i],
                                                            HPs["OffsetSpan"][i],
                                                            HPs["SystemChangeRate"],
                                                            HPs["MaxSystemVelocity"],
                                                            HPs["Duration"],
                                                            time)

        AnomalOffsetValues = getSingleParameterVariation(HPs["AnomalousOffset"][i],
                                                            HPs["AnomalousOffsetSpan"][i],
                                                            HPs["SystemChangeRate"],
                                                            HPs["MaxSystemVelocity"],
                                                            HPs["Duration"],
                                                            time)

        Offset = np.multiply(NormalOffsetValues,np.ones(len(BlendArray))-(BlendArray*HPs["AnomalyMagnitudeInOffset"])) + np.multiply(AnomalOffsetValues,BlendArray*HPs["AnomalyMagnitudeInOffset"])

 
        
        sumOfSines += Offset+ np.multiply(Amplitude,np.sin(np.Multiply(time,Frequency)))
        
        sunOfSines += (np.random(len(sumOfSines))*2*HPs["NoiseLevel"])-HPs["NoiseLevel"]

    return sumOfSines
    

def generateSet(**HPs,numSamples,containsAnomalies,dimensions):
 
    Time = np.linspace(0,HPs["TimeSpan"],int(HPs["TimeSpan"]/HPs["SampleRate"]))
    
    anomalyLenghInIndices = int(HPs["AnomalyDuration"]/HPs["SampleRate"])
    rampTime = int(HPs["AnomalyRampTime"]/HPs["SampleRate"])

    if anomalyLengthInIndices = 0:
        anomalyLengthInIndices = 1
    if rampTime = 0:
        rampTime = 1

    BlendArray = np.zeros(len(Time))
    IsAnomaly = np.zeros(len(Time))
    
    if(containsAnomalies):
        #Distirbute Anomalies
        for i in range(0,int(len(Time)/HPs["AnomalyChance"])):
            anomalyBegin = int(random()*(len(Time)-anomalyLengthInIndices))
            BlendArray[anomalyBegin:anomalyBegin+anomalyLengthInIndices] = 1
        #Smoothing for the ramptime:
        BlendArray = np.convolve(BlendArray,np.ones(rampTime),"same")
        IsAnomaly[np.argwhere(BlendArray > HPs["AnomalyThreshold"])] = 1
    
    data = pd.DataFrame.empty()
    data["Time"] = Time
    data["Is Anomaly"] = IsAnomaly

    for i in range(0,dimensions):
        data["Dimension "+str(i+1)] = generate1DSines(**HPs,BlendArray*HPs["AnomalyInDimension"],Time)

    DataSet,IsAnomaly = RandomSampling(data,numSamples)
    
    return DataBlock("Synthetic Sines",DataSet,dimensions,**HPs)

def generateData(dimnsions,**hyperParameters):

    #That Hyperparameter Set generates one Sine wave that varies in Frequency and 
    #in amplitude to the specified degree (1 center Amplitude, varies between 0.9 an 1.1 and so forth).
    #With a chance of AomalyChance, An anomaly is placed on the time series. That means that, if
    # ANomaly in frequency or amplitude is true for the given sine, the frequency or amplitude will be sampled
    # From the anomal paramters
    # The ramptime designates the time the system transitions from normal to anormal and back. At the points in the 
    # transition, a linear combination of normal an anormal is used.

    defaultHyperParameters = 
    {
        "Amplitudes" :          [1 ],#center of the amplitude parameter span
        "AmplitudeSpan" :       [0.1 ],#plus minus
        "AnomalousAmplitudes" :     [1], 
        "AnomalousAmplitudeSpan" :  [0.1], #Plus Minus
        
        "Frequency" :           [4],
        "FrequencySpan" :       [1], #Plus Minus
        "AnomalousFrequency" :      [7], 
        "AnomalousFrequencySpan" :  [1],#Plus Minus
        
        "Offset" :              [0],
        "OffsetSpan":           [0],
        "AnomalousOffset" : [0], #Plus Minus
        "AnomalousOffsetSpan" : [0], #Plus Minus

        "AnomalyMagnitudeInAmplitude" :      [0],
        "AnomalyMagnitudeInFrequency" :      [1],
        "AnomalyMagnitudeInOffset"    :      [0.1],
        "AnomalyInDimension" : [1],
        
        "NoiseLevel" :              0.02,
        "MaxSystemVelocity":           0.1, # The maximum rate at which the parameters change in parameter value per dimension per second.
        "SystemChangeRate": 0.2,

        "AnomaliesInTrainingdata" : False,
        "AnomaliesInValidationdata" :True,

        "AnomalyThreshold": 0.3,
        "AnomalyRampTime":          1,
        "DefaultAnomalyDuration":   2,
        "SampleTime":               0.5,
        "AnomalyChance":            0.01,
        "Duration":                 10000,#duration of the timeseries in the dataset 
        "RandomSeed":               1,


        "TrainingSetSize": 1000,
        "ValidationSetSize":100,
        "TestSetSize" : 30,
    }
    

    HPs = {**defaultHyperParameters,**hyperParameters}
    
    return generateSet(**HPs,HPs["TrainingSetSize"],HPs["AnomaliesInTrainingdata"],dimensions),generateSet(**HPs,HPs["ValidationSetSize"],HPs["AnomaliesInValidationdata"],dimensions),generateSet(**HPs,HPs["TestSetSize"],True,dimensions)
