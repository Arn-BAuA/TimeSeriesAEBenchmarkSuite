#!/bin/python

from Models.FeedForward import Model as FeedForwardAE
from Models.RecurrendAE import Model as LSTMAE
from SetWrappers.UCRArchive import loadData as DataSet
from DataGenerators.Sines import generateData as Sines
from Trainers.SingleInstanceTrainer import Trainer as OnlineTrainer

from Benchmark import benchmark,initializeDevice
from Evaluation.QuickOverview import plotOverview

pathToSave = "LSTM Sines Test"

device = initializeDevice()
Dimensions = 2 # Dataset dimensions


trainingSet,validationSet,testSet = Sines(Dimensions)
#trainingSet,validationSet,testSet = DataSet(Dimensions,DataSet = "UMD")

#model = FeedForwardAE(Dimensions,device,InputSize = trainingSet.Length())
model = LSTMAE(Dimensions,device,CellKind = "LSTM")

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

