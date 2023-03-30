#!/bin/python

from AEModels.FeedForward import Model as ModelClass
from SetWrappers.AirQualityUCI import loadData
from Trainers.SingleInstanceTrainer import Trainer
from Benchmark import benchmark,initializeDevice

device = initializeDevice()
Dimensions = 1 # Dataset dimensions


model = ModelClass(Dimensions,device)
trainingSet,validationSet,testSet = loadData(Dimensions)
trainer = Trainer(model,device)

benchmark(trainingSet,validationSet,testSet,model,trainer,n_epochs=150,pathToSave="",device = device)
