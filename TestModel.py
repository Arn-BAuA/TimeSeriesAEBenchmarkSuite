#!/bin/python

from Models.FeedForward import Model as FeedForwardAE
from Models.RecurrendAE import Model as LSTMAE
from Models.CNN_AE import Model as CNNAE
from Models.AttentionBasedAE import Model as AttentionModel

from SetWrappers.UCRArchive import loadData as UCRDataSet
from SetWrappers.SMD import loadData as SMDDataSet
from SetWrappers.ECGDataSet import loadData as ECGDataSet
from DataGenerators.Sines import generateData as Sines
from Trainers.SingleInstanceTrainer import Trainer as OnlineTrainer
from Trainers.BatchedTrainer import Trainer as BatchedTrainer

from Benchmark import benchmark,initializeDevice
from Evaluation.QuickOverview import plotOverview

pathToSave = "Results/DebugSession"

device = initializeDevice()
Dimensions = 2 # Dataset dimensions


#trainingSet,validationSet,testSet = Sines(Dimensions)
#trainingSet,validationSet,testSet = UCRDataSet(Dimensions,DataSet = "UMD")
#trainingSet,validationSet,testSet = ECGDataSet(Dimensions)
trainingSet,validationSet,testSet = SMDDataSet(Dimensions,nNormalDimensions=1)

#model = FeedForwardAE(Dimensions,device,InputSize = trainingSet.Length())
#model = LSTMAE(Dimensions,device,CellKind = "LSTM")
#model = CNNAE(Dimensions,device,InputSize = trainingSet.Length())
#
#model = CNNAE(Dimensions,device,hasFFTEncoder = True)
model = AttentionModel(Dimensions,device)

#trainer = OnlineTrainer(model,device)
trainer = BatchedTrainer(model,device)

resultFolder = benchmark(trainingSet,
          validationSet,
          testSet,
          model,
          trainer,
          n_epochs=20,
          pathToSave=pathToSave,
          device = device)

plotOverview(resultFolder)

