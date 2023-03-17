#!/bin/python

import sys
from Benchmark import benchmark

#replacing / with . for the python import later
#removing the .py at the end
dataSet = sys.argv[1].replace("/",".")[:-3]
model = sys.argv[2].replace("/",".")
trainer = sys.argv[3].replace("/",".")

dataSet = __import__(dataSet+".load_data")
model = __import__(model+".Model")
trainer = __import__(trainer+".Trainer")

