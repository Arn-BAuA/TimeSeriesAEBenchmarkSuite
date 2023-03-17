#!/bin/python

from AEModels.FeedForward import Model as ModelClass
from SetWrappers.AirQualityUCI import loadData as dataSet
from Trainers.SingleInstanceTrainer import Trainer
from Benchmark import benchmark,getDevice



dimension = 1 
device = getDevice()

model = ModelClass(dimension,device)

benchmark(dataSet,{},model,{},Trainer,{},150,1,pathToSave="")
