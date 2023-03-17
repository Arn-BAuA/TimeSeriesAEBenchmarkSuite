#!/bin/python

import sys
from Benchmark import benchmark,getDevice

dimension = 1 
device = getDevice()

#replacing / with . for the python import later
#removing the .py at the end
dataSet = sys.argv[1].replace("/",".")[:-3]
model = sys.argv[2].replace("/",".")
trainer = sys.argv[3].replace("/",".")

dataSet = __import__(dataSet+".load_data")
ModelClass = __import__(model+".Model")
trainer = __import__(trainer+".Trainer")


model = ModelClass(dimension,device)

benchmark(dataSet,{},model,{},trainer,{},150,pathToSave="")
