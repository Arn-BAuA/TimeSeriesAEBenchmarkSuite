#!/bin/python

from AEModels.FeedForward import Model as ModelClass

#from SetWrappers.AirQualityUCI import loadData
from DataGenerators.Sines import generateData as loadData

from Trainers.SingleInstanceTrainer import Trainer
from Benchmark import benchmark,initializeDevice
import sys
from Evaluation.QuickOverview import plotOverview

if len(sys.argv) == 1:
    pathToSave = input("Pleas specify a foldername for the testrun.")
else:
    pathToSave = sys.argv[1]

device = initializeDevice()
Dimensions = 2 # Dataset dimensions

trainingSet,validationSet,testSet = loadData(Dimensions)

model = ModelClass(Dimensions,device)


trainer = Trainer(model,device)

resultFolder = benchmark(trainingSet,
                validationSet,
                testSet,
                model,
                trainer,
                n_epochs=20,
                pathToSave=pathToSave,
                device = device)

plotOverview(resultFolder)

