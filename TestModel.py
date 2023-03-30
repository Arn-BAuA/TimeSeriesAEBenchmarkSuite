#!/bin/python

from AEModels.FeedForward import Model as ModelClass
from SetWrappers.AirQualityUCI import loadData
from Trainers.SingleInstanceTrainer import Trainer
from Benchmark import benchmark,initializeDevice
import sys
from Evaluation.QuickOverview import plotOverview

if len(sys.argv) == 1:
    pathToSave = input("Pleas specify a foldername for the testrun.")
else:
    pathToSave = sys.argv[1]

device = initializeDevice()
Dimensions = 1 # Dataset dimensions


model = ModelClass(Dimensions,device)
trainingSet,validationSet,testSet = loadData(Dimensions)
trainer = Trainer(model,device)

benchmark(trainingSet,
          validationSet,
          testSet,
          model,
          trainer,
          n_epochs=100,
          pathToSave=pathToSave,
          device = device)

plotOverview("Results/"+pathToSave)
