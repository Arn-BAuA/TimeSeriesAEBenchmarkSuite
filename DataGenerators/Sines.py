#!/bin/python

################################################
# This is a Datagenerator for multivariate timeseries data, consisting of multiple sine waves added together in every dimension.
# The data generator can also includes Anomalies in Amplitude, Frequency and Offset of all the sines, or single sines in the sries.
# The generationprocess looks as follows:
# The User specifies normal an anomal parameters for the sines in the series. For each parameter, a center and a span is specified.
# The Generator than calculates a random walk in the intervall that is specified for each normal and anomal parameter.
# The speed of the random walk can also be choosen by the user.
# Quick summary: For each parameter and each point in time we now have a random walk for the normal state of the parameter and the anomal state.
# Now it is choosen at random, by a rate defined by the user, where anomalies will be. At the points where anomalies are,
# the parameter random walks are blended, with a smooth transition of a duration that is defined by the user.
# Thus we get one random walk set of parameters, that contains the anomalies.
# This random walk than gets put into the equation of sines (A*sin(t\phi+\theta)+B) to generate the time series
# After the generation of the data, white noise is added at a level specified by the user.
# Points in time are always equidistant.

from BlockAndDatablock import DataBlock
import numpy as np
from numpy.random import random
import pandas as pd
from Utility.DataSetSampler import RandomSampling

# Random walk for one INtervall / one Parameter
def getSingleParameterVariation(value, #center of the interval of possible values
                                span, #span of the intervall of possible values
                                changeRate,#Rate that determines when the random walk changes direction
                                MaxSystemVelocity, #max speed of the random walk in parameter per time
                                duration, #duration of the simulated time series
                                time #the time (array of time values)
                                ):
    nChanges = int(duration*changeRate)

    timePoints = random([nChanges])*duration #points in time where change occures
    timePoints = np.sort(timePoints)
    timePoints[0]=0
    timePoints[-1] = duration

    timeIntervals = timePoints[1:] - timePoints[:-1] #intervals where the direction of random walk is constant
    
    ##########################################
    #THe next section looks a bit complicated. This is to assure, that we don't walk out of the interval
    # during the walk while keeping the velocity.
    #
    possibleMaxChange = timeIntervals*MaxSystemVelocity 
    randomWalkValues = np.zeros(nChanges)

    #Random Walk:
    randomWalkValues[0] = (random()*2*span)-span #initial Parameter

    change = np.multiply(2*possibleMaxChange,random([nChanges-1]))-possibleMaxChange

    #here its assured that we dont walk out of the intervall:
    for i in range(1,len(randomWalkValues)):
        
        if np.abs(change[i-1]) >= span:
            #We walked so long, that we covered the whole span of the intervall.
            # in this case, a new location for our parameter is sampled at random
            newParameter = (2*span*random())-span
        else:
            newParameter = randomWalkValues[i-1] + change[i-1]

            #We would have walked out, instead, we pretend that we "bounced" off the intervall boundary
            if newParameter < -span:
                newParameter = -span -(newParameter+span)
            if newParameter > span:
                mewParameter = span - (newParameter-span)
            
        randomWalkValues[i] = newParameter
    
    # We now have the position of the parameter at the points where the change occures.
    # We now transfer this to the points in time of the time series by linear interpolation.
    parameterValues = np.zeros(len(time))

    for i in range(0,len(time)):
        t1Index = np.argwhere(timePoints <= time[i]).max() #Since they are sorted, returns the index of the largest t value smaller time[i]
        t2Index = np.argwhere(timePoints >= time[i]).min()

        t1 = timePoints[t1Index]
        v1 = randomWalkValues[t1Index]
        t2 = timePoints[t2Index]
        v2 = randomWalkValues[t2Index]
       
        
        if(t1 == t2):
            parameterValues[i] = v1
        else:
            #LinearInterpolation:
            parameterValues[i] = v1*((t2-time[i])/(t2-t1)) + v2 * ((time[i]-t1)/(t2-t1))

        
     
    retVal = parameterValues+value
    return retVal

def generate1DSines(dimension,BlendArray,time,**HPs):
    
    sumOfSines = np.zeros(len(time))

    for i in range(0,len(HPs["Amplitudes"][dimension])):
        
        #Amplitude
        NormalAmplitudeValues = getSingleParameterVariation(HPs["Amplitudes"][dimension][i],
                                                            HPs["AmplitudeSpan"][dimension][i],
                                                            HPs["SystemChangeRate"],
                                                            HPs["MaxSystemVelocity"],
                                                            HPs["Duration"],
                                                            time)

        AnomalAmplitudeValues = getSingleParameterVariation(HPs["AnomalousAmplitudes"][dimension][i],
                                                            HPs["AnomalousAmplitudeSpan"][dimension][i],
                                                            HPs["SystemChangeRate"],
                                                            HPs["MaxSystemVelocity"],
                                                            HPs["Duration"],
                                                            time)

        Amplitude = np.multiply(NormalAmplitudeValues,np.ones(len(BlendArray))-(BlendArray*HPs["AnomalyMagnitudeInAmplitude"][dimension][i])) + np.multiply(AnomalAmplitudeValues,BlendArray*HPs["AnomalyMagnitudeInAmplitude"][dimension][i])

        #Frequency
        NormalFrequencyValues = getSingleParameterVariation(HPs["Frequency"][dimension][i],
                                                            HPs["FrequencySpan"][dimension][i],
                                                            HPs["SystemChangeRate"],
                                                            HPs["MaxSystemVelocity"],
                                                            HPs["Duration"],
                                                            time)

        AnomalFrequencyValues = getSingleParameterVariation(HPs["AnomalousFrequency"][dimension][i],
                                                            HPs["AnomalousFrequencySpan"][dimension][i],
                                                            HPs["SystemChangeRate"],
                                                            HPs["MaxSystemVelocity"],
                                                            HPs["Duration"],
                                                            time)

        Frequency = np.multiply(NormalFrequencyValues,np.ones(len(BlendArray))-(BlendArray*HPs["AnomalyMagnitudeInFrequency"][dimension][i])) + np.multiply(AnomalFrequencyValues,BlendArray*HPs["AnomalyMagnitudeInFrequency"][dimension][i])
        


        #Offset
        NormalOffsetValues = getSingleParameterVariation(HPs["Offset"][dimension][i],
                                                            HPs["OffsetSpan"][dimension][i],
                                                            HPs["SystemChangeRate"],
                                                            HPs["MaxSystemVelocity"],
                                                            HPs["Duration"],
                                                            time)

        AnomalOffsetValues = getSingleParameterVariation(HPs["AnomalousOffset"][dimension][i],
                                                            HPs["AnomalousOffsetSpan"][dimension][i],
                                                            HPs["SystemChangeRate"],
                                                            HPs["MaxSystemVelocity"],
                                                            HPs["Duration"],
                                                            time)

        Offset = np.multiply(NormalOffsetValues,np.ones(len(BlendArray))-(BlendArray*HPs["AnomalyMagnitudeInOffset"][dimension][i])) + np.multiply(AnomalOffsetValues,BlendArray*HPs["AnomalyMagnitudeInOffset"][dimension][i])

 
        
        if(HPs["ContinousFrequencyBlending"]):
            timeDelta = np.zeros(len(time)) 
            timeDelta[1:] = time[1:]-time[:-1]
            advance = np.multiply(timeDelta,Frequency)
            #Adding the advance together to get an argument for the sine
            tTimesf = np.tril(np.ones([len(advance),len(advance)])).dot(advance)

            sumOfSines += Offset+ np.multiply(Amplitude,np.sin(tTimesf))
        else:
            sumOfSines += Offset+ np.multiply(Amplitude,np.sin(np.multiply(time,Frequency)))
        
        sumOfSines += (random(len(sumOfSines))*2*HPs["NoiseLevel"])-HPs["NoiseLevel"]
    
    return sumOfSines
    
def generateSet(numSamples,containsAnomalies,dimensions,**HPs):
    
    Time = np.linspace(0,HPs["Duration"],int(HPs["Duration"]/HPs["SampleTime"]))
    
    anomalyLengthInIndices = int(HPs["AnomalyDuration"]/HPs["SampleTime"])
    rampTime = int(HPs["AnomalyRampTime"]/HPs["SampleTime"])

    if anomalyLengthInIndices == 0:
        anomalyLengthInIndices = 1
    if rampTime == 0:
        rampTime = 1

    BlendArray = np.zeros(len(Time))
    IsAnomaly = np.zeros(len(Time))
    
    if(containsAnomalies):
        #Distirbute Anomalies
        for i in range(0,int(HPs["Duration"]*HPs["AnomalyChance"])):
            anomalyBegin = int(random()*(len(Time)-anomalyLengthInIndices))
            BlendArray[anomalyBegin:anomalyBegin+anomalyLengthInIndices] = 1
        #Smoothing for the ramptime:
        BlendArray = np.convolve(BlendArray,np.ones(rampTime),"same")
        IsAnomaly[np.argwhere(BlendArray > HPs["AnomalyThreshold"])] = 1
    
    data = pd.DataFrame()
    data["Time"] = Time
    data["Is Anomaly"] = IsAnomaly
    
    for i in range(0,dimensions):
        data["Dimension "+str(i+1)] = generate1DSines(i,BlendArray*HPs["AnomalyInDimension"][i],Time,**HPs)

    DataSet,IsAnomaly = RandomSampling(data,numSamples,HPs["SampleWindowSize"],includeTime = False,dateTimeColumn = "Time")
   
    Block = DataBlock("Synthetic Sines",DataSet,dimensions,**HPs)
    Block.setLabels(IsAnomaly)
    return Block

def generateData(dimensions,**hyperParameters):
    


    defaultHyperParameters = {
        "Amplitudes" :          [[1],[1] ],#center of the amplitude parameter span
        "AmplitudeSpan" :       [[0.2],[0.1] ],#plus minus
        "AnomalousAmplitudes" :     [[1.6],[1.6]], 
        "AnomalousAmplitudeSpan" :  [[0],[0]], #Plus Minus
        
        "Frequency" :           [[0.2],[0.3]],
        "FrequencySpan" :       [[0],[0]], #Plus Minus
        "AnomalousFrequency" :      [[0.5],[0.5]], 
        "AnomalousFrequencySpan" :  [[0],[0]],#Plus Minus
        
        "Offset" :              [[0],[0]],
        "OffsetSpan":           [[0],[0]],
        "AnomalousOffset" : [[0],[0]], #Plus Minus
        "AnomalousOffsetSpan" : [[0],[0]], #Plus Minus

        "AnomalyMagnitudeInAmplitude" :      [[1],[1]],
        "AnomalyMagnitudeInFrequency" :      [[0.1],[0.1]],
        "AnomalyMagnitudeInOffset"    :      [[0.1],[0.1]],
        "AnomalyInDimension" : [[1],[1]],
        
        "NoiseLevel" :              0.02,
        "MaxSystemVelocity":           0.01, # The maximum rate at which the parameters change in parameter value per dimension per second.
        "SystemChangeRate": 0.01,

        "AnomaliesInTrainingdata" : False,
        "AnomaliesInValidationdata" :True,
        
        "ContinousFrequencyBlending":True, #..

        "AnomalyThreshold": 0.3,
        "AnomalyRampTime":          1,
        "AnomalyDuration":   2,
        "SampleTime":               0.2,
        "AnomalyChance":            0.006,
        "Duration":                 2000,#duration of the timeseries in the dataset 
        "RandomSeed":               1,


        "TrainingSetSize": 200,
        "ValidationSetSize":50,
        "TestSetSize" : 30,
        "SampleWindowSize":150,
    }
    

    HPs = {**defaultHyperParameters,**hyperParameters}
    
    return generateSet(HPs["TrainingSetSize"],HPs["AnomaliesInTrainingdata"],dimensions,**HPs),generateSet(HPs["ValidationSetSize"],HPs["AnomaliesInValidationdata"],dimensions,**HPs),generateSet(HPs["TestSetSize"],True,dimensions,**HPs)


