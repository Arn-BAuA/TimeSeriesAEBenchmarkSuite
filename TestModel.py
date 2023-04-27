#!/bin/python

from AEModels.FeedForward import Model as FeedForwardAE
from SetWrappers.UCRArchive import loadData as DataSet
from Trainers.SingleInstanceTrainer import Trainer as OnlineTrainer

from Benchmark import benchmark,initializeDevice
from Evaluation.QuickOverview import plotOverview

pathToSave = "UCR Set Demo"

device = initializeDevice()
Dimensions = 2 # Dataset dimensions


trainingSet,validationSet,testSet = DataSet(Dimensions,DataSet = "UMD")

model = FeedForwardAE(Dimensions,device,InputSize = trainingSet.Length())

trainer = OnlineTrainer(model,device)

resultFolder = benchmark(trainingSet,
          validationSet,
          testSet,
          model,
          trainer,
          n_epochs=40,
          pathToSave=pathToSave,
          device = device)

plotOverview(resultFolder)

