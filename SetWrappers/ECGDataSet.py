# Set Wraper for any of the datasets in the UCR Archive (https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
# The 

import pandas as pd
from random import random
import numpy as np
import torch
from BlockAndDatablock import DataBlock
from Utility.DataSetSampler import fromClassification2AnoDetect as sampleDataSet

def getDatasetsInArchive():
    return UCRDatasets

import os
dirname = os.path.dirname(__file__)
ECGPath = os.path.join(dirname,"../data/ECG/")

def loadData(dimensions,**hyperParameters):

    defaultHyperParameters = {
            "ArrythmiaNormals":[0],#Classes from arrythmia that are used as normals, none, if left emty
            "ArrythmiaAnomalys":[1,2,3,4],#classes from arrythmia that are used as anomalies, none if left empty
            "PTBNormals":[],#Classes from PTB that are concidered as normal, none if left empty
            "PTBAnomalys":[],#Classes from the PTB that are used as anomalies, none if left empty
            "AnomalyPercentageTrain":0,#Percentage of anomalies in training set
            "AnomalyPercentageValidation":10, #percentage of anomalies in validation set
            "AnomalyPercentageTest":10,# percentage of anomalies in test set
            "SameClassForAllDimensions":False,#when true: All dimensions are of the same class, else, they are random
            "AllDimensionsAnomal":False,#If this is true and the value is an anomaly, all the dimensions are an anomaly
            "nAnomalDimensions":1,#if it is an anomaly: How many dimensions are anomal
            "UseArythmiaSplit":True,#For the used arythmia data: Use the split for training and test like in the original csv files:
            "PercentTrainingSet":70,# The percentage of samples that is split off for the training.
            "PercentValidationSet":20, # The percentage of samples that is split off for the validation.
            "PercentTestSet":10, # The percentage of samples that is split off for th testing.
            "TrainingSetSize":400,
            "ValidationSetSize":100,
            "TestSetSize":30,
        }
    HPs={**defaultHyperParameters,**hyperParameters}
    
    #trainingData = pd.read_csv(ECGPath+HPs["DataSet"]+"/"+HPs["DataSet"]+"_TRAIN.tsv",sep='\t',header=None)
    #testData = pd.read_csv(UCRPath+HPs["DataSet"]+"/"+HPs["DataSet"]+"_TEST.tsv",sep='\t',header=None)
    
    #Loading sets
    arrythmiaTest =
    arrythmiaTrain = 
    ptbNormal = 
    ptbAnomal =

    trainingDataNormal =
    trainingDataAnomal =
    validationDataNormal =
    validationDataAnomal =
    testDataNormal =
    testDataAnormal = 


    

    
    
    

    #droping the first column with the labels
    trainingData = trainingData.drop(trainingData.columns[-1],axis=1)
    trainingAnomaly = trainingAnomaly.drop(trainingAnomaly.columns[-1],axis=1)
    testData = testData.drop(testData.columns[-1],axis=1)
    testAnomaly = testAnomaly.drop(testAnomaly.columns[-1],axis=1)

    trainingSet = [0]*HPs["TrainingSetSize"]
    trainingAnomalyIndex = [0]*HPs["TrainingSetSize"]
    for i in range(0,HPs["TrainingSetSize"]):
        trainingSet[i],trainingAnomalyIndex[i] = sampleDataSet(dimensions,
                                       trainingData,
                                       trainingAnomaly,
                                       HPs["AnomalyPercentageTrain"],
                                       HPs["SameClassForAllDimensions"],
                                       HPs["nAnomalDimensions"],
                                       HPs["AllDimensionsAnomal"]) 
    trainingBlock = DataBlock("UCR Archive - "+HPs["DataSet"],trainingSet,dimensions,**HPs)
    trainingBlock.setLabels(trainingAnomalyIndex)
    trainingBlock.setGeneratedFromCDS(True)

    validationSet = [0]*HPs["ValidationSetSize"]
    validationAnomalyIndex = [0]*HPs["ValidationSetSize"]
    for i in range(0,HPs["ValidationSetSize"]):
        validationSet[i],validationAnomalyIndex[i] = sampleDataSet(dimensions,
                                       trainingData,
                                       trainingAnomaly,
                                       HPs["AnomalyPercentageValidation"],
                                       HPs["SameClassForAllDimensions"],
                                       HPs["nAnomalDimensions"],
                                       HPs["AllDimensionsAnomal"]) 
    
    validationBlock = DataBlock("UCR Archive - "+HPs["DataSet"],validationSet,dimensions,**HPs)
    validationBlock.setLabels(validationAnomalyIndex)
    validationBlock.setGeneratedFromCDS(True)

    testSet = [0]*HPs["TestSetSize"]
    testAnomalyIndex = [0]*HPs["TestSetSize"]
    for i in range(0,HPs["TestSetSize"]):
        testSet[i],testAnomalyIndex[i] = sampleDataSet(dimensions,
                                       testData,
                                       testAnomaly,
                                       HPs["AnomalyPercentageTest"],
                                       HPs["SameClassForAllDimensions"],
                                       HPs["nAnomalDimensions"],
                                       HPs["AllDimensionsAnomal"]) 

    testBlock = DataBlock("UCR Archive - "+HPs["DataSet"],testSet,dimensions,**HPs)
    testBlock.setLabels(testAnomalyIndex)
    testBlock.setGeneratedFromCDS(True)
        
    return trainingBlock,validationBlock,testBlock

UCRDatasets = [
            "ACSF1",
            "Adiac",
            "AllGestureWiimoteX",
            "AllGestureWiimoteY",
            "AllGestureWiimoteZ",
            "ArrowHead",
            "Beef",
            "BeetleFly",
            "BirdChicken",
            "BME",
            "Car",
            "CBF",
            "Chinatown",
            "ChlorineConcentration",
            "CinCECGTorso",
            "Coffee",
            "Computers",
            "CricketX",
            "CricketY",
            "CricketZ",
            "Crop",
            "DiatomSizeReduction",
            "DistalPhalanxOutlineAgeGroup",
            "DistalPhalanxOutlineCorrect",
            "DistalPhalanxTW",
            "DodgerLoopDay",
            "DodgerLoopGame",
            "DodgerLoopWeekend",
            "Earthquakes",
            "ECG200",
            "ECG5000",
            "ECGFiveDays",
            "ElectricDevices",
            "EOGHorizontalSignal",
            "EOGVerticalSignal",
            "EthanolLevel",
            "FaceAll",
            "FaceFour",
            "FacesUCR",
            "FiftyWords",
            "Fish",
            "FordA",
            "FordB",
            "FreezerRegularTrain",
            "FreezerSmallTrain",
            "Fungi",
            "GestureMidAirD1",
            "GestureMidAirD2",
            "GestureMidAirD3",
            "GesturePebbleZ1",
            "GesturePebbleZ2",
            "GunPoint",
            "GunPointAgeSpan",
            "GunPointMaleVersusFemale",
            "GunPointOldVersusYoung",
            "Ham",
            "HandOutlines",
            "Haptics",
            "Herring",
            "HouseTwenty",
            "InlineSkate",
            "InsectEPGRegularTrain",
            "InsectEPGSmallTrain",
            "InsectWingbeatSound",
            "ItalyPowerDemand",
            "LargeKitchenAppliances",
            "Lightning2",
            "Lightning7",
            "Mallat",
            "Meat",
            "MedicalImages",
            "MelbournePedestrian",
            "MiddlePhalanxOutlineAgeGroup",
            "MiddlePhalanxOutlineCorrect",
            "MiddlePhalanxTW",
            "MixedShapesRegularTrain",
            "MixedShapesSmallTrain",
            "MoteStrain",
            "NonInvasiveFetalECGThorax1",
            "NonInvasiveFetalECGThorax2",
            "OliveOil",
            "OSULeaf",
            "PhalangesOutlinesCorrect",
            "Phoneme",
            "PickupGestureWiimoteZ",
            "PigAirwayPressure",
            "PigArtPressure",
            "PigCVP",
            "PLAID",
            "Plane",
            "PowerCons",
            "ProximalPhalanxOutlineAgeGroup",
            "ProximalPhalanxOutlineCorrect",
            "ProximalPhalanxTW",
            "RefrigerationDevices",
            "Rock",
            "ScreenType",
            "SemgHandGenderCh2",
            "SemgHandMovementCh2",
            "SemgHandSubjectCh2",
            "ShakeGestureWiimoteZ",
            "ShapeletSim",
            "ShapesAll",
            "SmallKitchenAppliances",
            "SmoothSubspace",
            "SonyAIBORobotSurface1",
            "SonyAIBORobotSurface2",
            "StarLightCurves",
            "Strawberry",
            "SwedishLeaf",
            "Symbols",
            "SyntheticControl",
            "ToeSegmentation1",
            "ToeSegmentation2",
            "Trace",
            "TwoLeadECG",
            "TwoPatterns",
            "UMD",
            "UWaveGestureLibraryAll",
            "UWaveGestureLibraryX",
            "UWaveGestureLibraryY",
            "UWaveGestureLibraryZ",
            "Wafer",
            "Wine",
            "WordSynonyms",
            "Worms",
            "WormsTwoClass",
            "Yoga",
    ]


