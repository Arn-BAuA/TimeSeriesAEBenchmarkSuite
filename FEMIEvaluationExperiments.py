#A File Containing the Experiments to Evaluate the FEMI-Index

from Benchmark import benchmark,initializeDevice
from FEMIDataIndex import computeFEMIIndex
import numpy as np
import json

def SinesExperiment():

    FEMI = {}

    from Trainers.BatchedTrainer import Trainer as BatchedTrainer
    from Models.FeedForward import Model as FeedForwardAE
    from DataGenerators.Sines import generateData as Sines
    
    Dimensions=2

    device = initializeDevice()
    
    path = "FEMISineExperiment/"

    for AnomalyAmplitude in np.linspace(1,2,11):
        trainingSet,validationSet,testSet = Sines(Dimensions,
                                                  AnomalousFrequency = [[0.2],[0.1]],
                                                  AnmalousAmplitudes =[[AnomalyAmplitude],[AnomalyAmplitude]],
                                                TestSetSize=200
                                                  )
        

        model = FeedForwardAE(Dimensions,device,InputSize = trainingSet.Length())

        trainer = BatchedTrainer(model,device)

        resultFolder = benchmark(trainingSet,
            validationSet,
            testSet,
            model,
            trainer,
            n_epochs=10,
            pathToSave=path+"AnomalyAmplitude"+str(AnomalyAmplitude),
            device = device)

        E_polar,MI_polar = computeFEMIIndex(trainingSet,testSet)
        E,MI = computeFEMIIndex(trainingSet,testSet,False)
        FEMI[AnomalyAmplitude] ={"E":E,"MI":MI,"E_Polar":E_polar,"MI_Polar":MI_polar}

    
        with open("Results/"+path+"FEMIIndices.json","w") as f:
            json.dump(FEMI,f,default=str,indent=4)

def UCRExperiment():

    FEMI = {}

    from Trainers.BatchedTrainer import Trainer as BatchedTrainer
    from Models.FeedForward import Model as FeedForwardAE
    from Models.CNN_AE import Model as CNNAE

    from SetWrappers.UCRArchive import loadData as DataSet
    from SetWrappers.UCRArchive import getDatasetsInArchive
    
    Dimensions=1

    device = initializeDevice()
    
    path = "FEMIUCRExperiment/"

    dataSets = getDatasetsInArchive()
    
    for dsName in dataSets:
        print(dsName+":")

        trainingSet,validationSet,testSet = DataSet(
                                                DataSet = dsName,
                                                dimensions=Dimensions,
                                                TrainingSetSize = 100,
                                                ValidationSetSize = 100,
                                                TestSetSize = 10,
                                                anomalyPercentageTest = 10)


        model = CNNAE(Dimensions,device,InputSize = trainingSet.Length())
        #model = FeedForwardAE(Dimensions,device,InputSize = trainingSet.Length())

        trainer = BatchedTrainer(model,device)

        resultFolder = benchmark(trainingSet,
            validationSet,
            testSet,
            model,
            trainer,
            n_epochs=60,
            pathToSave=path+dsName,
            device = device)

        E_polar,MI_polar = computeFEMIIndex(trainingSet,validationSet)
        E,MI = computeFEMIIndex(trainingSet,validationSet,False)
        FEMI[dsName] ={"E":E,"MI":MI,"E_Polar":E_polar,"MI_Polar":MI_polar}

    
        with open("Results/"+path+"FEMIIndices.json","w") as f:
            json.dump(FEMI,f,default=str,indent=4)

#UCRExperiment()
#######################################################
#           Code for Plotting the Experiment          #
#######################################################

path = "Results/FEMIUCRExperiment/"

f = open(path+"FEMIIndices.json","r")
FEMIIndices = json.load(f)
f.close()

import pandas as pd
import matplotlib.pyplot as plt

#relevantEpochs = [0,10,20,30,40,50,60]
relevantEpochs = [0,60]

fig,axs = plt.subplots(len(relevantEpochs),2)

for key in FEMIIndices:
    
    data = pd.read_csv(path+key+"/Errors.csv",sep="\t")
    
    AUCScore = []

    for e in relevantEpochs:
        AUCScore.append(data.loc[data["#Epoch"] == e]["AUC Score on Validation Set"])
    
    FEMIIndices[key]["AUC"] = AUCScore

colorMap = plt.cm.plasma

for i,epoch in enumerate(relevantEpochs):

    for key in FEMIIndices:
    
        axs[i,0].scatter([FEMIIndices[key]["E"]],[FEMIIndices[key]["MI"]],color=colorMap(FEMIIndices[key]["AUC"][i]))
        axs[i,1].scatter([FEMIIndices[key]["E_Polar"]],[FEMIIndices[key]["MI_Polar"]],color=colorMap(FEMIIndices[key]["AUC"][i]))

    axs[i,0].set_title("Component FEMI Indices @ Epoch "+str(epoch))
    axs[i,0].set_xlabel("E")
    axs[i,0].set_ylabel("MI")

    axs[i,1].set_title("Polar FEMI Indices @ Epoch "+str(epoch))
    axs[i,1].set_xlabel("E")
    axs[i,1].set_ylabel("MI")
    

fig.tight_layout()
plt.show()
plt.close()

#SinesExperiment()


