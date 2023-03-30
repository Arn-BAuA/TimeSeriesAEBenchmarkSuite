#!/bin/python

################################################
# THis is a Datagenerator, that adds a bunch of sine waves with noise and drift.


from BlockAndDatablock import Datablock
import numpy as np
from numpy.random import random

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

def generate1DSines(**HPs):
    
    numSines = len(HPs["SineAmplitudes"])

    
    
    

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
        "SineAmplitudes" :          [1 ],#center of the amplitude parameter span
        "SineAmplitudeSpan" :       [0.1 ],#plus minus
        "SineFrequency" :           [4],
        "SineFrequencySpan" :       [1], #Plus Minus
        "SineOffset" :              [0],
        "SineOffsetSpan":           [0],

        "AnomalousAmplitudes" :     [1], 
        "AnomalousAmplitudeSpan" :  [0.1], #Plus Minus
        "AnomalousFrequency" :      [7], 
        "AnomalousFrequencySpan" :  [1],#Plus Minus

        "AnomalyInAmplitude" :      [False],
        "AnomalyInFrequency" :      [True],
        "AnomalyInOffset"    :      [False],
        
        "NoiseLevel" :              0.02,
        "MaxSystemVelocity":           0.1, # The maximum rate at which the parameters change in parameter value per dimension per second.
        "SystemChangeRate": 0.2,

        "AnomalyRampTime":          1,
        "DefaultAnomalyDuration":   2,
        "SampleTime":               0.5,
        "AnomalyChance":            0.01,
        "TimeSpan":                 10000,#duration of the timeseries in the dataset 
        "RandomSeed":               1,
    }
    
    HPs = {**defaultHyperParameters,**hyperParameters}
    
    
