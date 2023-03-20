#!/bin/python

from AEModels.FeedForward import Model as ModelClass
from SetWrappers.AirQualityUCI import loadData
from Trainers.SingleInstanceTrainer import Trainer
from Benchmark import benchmark,initGlobalEnvironment

initGlobalEnvironment(DataSetDimensions=1)

model = ModelClass()

trainingSet,validationSet,testSet = loadData()

benchmark(trainingSet,validationSet,testSet,model,Trainer,n_epochs=150,pathToSave="")
