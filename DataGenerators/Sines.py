#!/bin/python

################################################
# THis is a Datagenerator, that adds a bunch of sine waves with noise and drift.


from BlockAndDatablock import Datablock
import numpy as np

def generate1DSines(**HPs):

    

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

        "AnomalousAmplitudes" :     [1], 
        "AnomalousAmplitudeSpan" :  [0.1], #Plus Minus
        "AnomalousFrequency" :      [7], 
        "AnomalousFrequencySpan" :  [1],#Plus Minus

        "AnomalyInAmplitude" :      [False],
        "AnomalyInFrequency" :      [True],
        
        "NoiseLevel" :              0.02,
        "SystemVelocity":           0.1, # The maximum rate at which the parameters change in parameter value per dimension per second.
        "AnomalyRampTime":          1,
        "DefaultAnomalyDuration":   2,
        "SampleTime":               0.5,
        "AnomalyChance":            0.01,
        "TimeSpan":                 10000,#duration of the timeseries in the dataset 
        "RandomSeed":               1,
    }
    
    HPs = {**defaultHyperParameters,**hyperParameters}
    
    
