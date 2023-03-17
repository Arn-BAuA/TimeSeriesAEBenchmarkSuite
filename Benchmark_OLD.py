#!/bin/python

import pandas as pd
from datetime import date,time,datetime
import matplotlib.pyplot as plt

PathToAirqualityData = "data/AirQualityUCI.xlsx"
Columns = [
        "PT08.S1(CO)",
        "PT08.S2(NMHC)",
        "PT08.S3(NOx)",
        "PT08.S4(NO2)",
        "PT08.S5(O3)",
        ]

dimensions = 1
sampleWindowSize = 150 # Number of samples in one Window for Training / testing
useTimestapmsAsInput = False
n_epochs = 70 #for Training

#################################
# Data Loading                  #
#################################

relevantColumns = Columns[0:dimensions]

#loading and preparing air quality data
def loadAirQualityData():
    dataset = pd.read_excel(PathToAirqualityData,parse_dates=[["Date","Time"]])
    dataset = dataset[["Date_Time"]+relevantColumns]
    dataset[relevantColumns] = (dataset[relevantColumns]-dataset[relevantColumns].mean())/dataset[relevantColumns].std()
    return dataset

allData = loadAirQualityData()

#allData.plot(x="Date_Time",y=relevantColumns)
#print(allData)
#plt.show()

#################################
# Sampling the DataSets         #
#################################

from random import random, seed
import numpy as np
import torch

seed(1)

####################
# Take snippets at random locations from the dataset
#
#
def SampleDataSet(beginDate,endDate,numberOfSamples):
    
    DataSet = [0] * numberOfSamples
    sampleArea = allData.loc[(allData["Date_Time"]>beginDate) & (allData["Date_Time"]<= endDate)]
   
    if not useTimestapmsAsInput:
        sampleArea = sampleArea.drop(columns=["Date_Time"])
    else:
        #conversion of datetime to timestamp for later conversion to pytorch tensor
        sampleArea["Date_Time"] = sampleArea.Date_Time.values.astype(np.int64)

    if len(sampleArea.index) < sampleWindowSize:
        raise Exception(f"The Samplewindow size is larger than the given range to sample from ({sampleWindowSize}/{len(sampleArea.index)})")

    #Bogo (Random) Sampling...
    for i in range(0,numberOfSamples):
        
        position =int(random() * float(len(sampleArea.index)-sampleWindowSize))
        #sampling
        sequence = sampleArea.iloc[np.arange(position,position+sampleWindowSize)]
        #conversion to tensor
        DataSet[i] = torch.tensor(sequence.values.astype(np.float32))
        #DataSet[i] = torch.transpose(DataSet[i],0,1)
    
    return DataSet

trainingSet = SampleDataSet(datetime(2004,4,1),datetime(2005,1,1),1000)
validationSet = SampleDataSet(datetime(2005,1,1),datetime(2005,3,1),100)
testSet = SampleDataSet(datetime(2005,3,1),datetime(2005,4,1),30)

#print(trainingSet[0])
#print(trainingSet.size())

########################################################################
#               The Model                                              #
########################################################################

from torch import nn

device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
else: 
    if torch.backends.mps.is_available():
        device = "mps"

class BinCoder(nn.Module):#Autoencoder form Bin's drift detection script. (Malhorta et al)
    
    def __init__(self,input_dim,hidden_dim,num_layers=4):

        super().__init__()
        self.input_dim,self.hidden_dim = input_dim,hidden_dim

        self.encoder = nn.LSTM(
                    input_size = self.input_dim,
                    hidden_size = self.hidden_dim,
                    num_layers = num_layers,
                    batch_first = True
                )

        self.decoder = nn.LSTM(
                    input_size = self.input_dim,
                    hidden_size = self.hidden_dim,
                    num_layers = num_layers,
                    batch_first = True
                )
        
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

        self.output = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self,x):
        
        encoder_out,encoder_hidden = self.encoder(x)

        decoder_input = torch.zeros_like(x)
        
        decoder_output,decoder_hidden = self.decoder(decoder_input,encoder_hidden)
        reconstruction = self.output(decoder_output)
        reconstruction = torch.flip(reconstruction,[1])
        return reconstruction

class ArnCoder(nn.Module): #Plain Feed Forward Encoder....
    
    def __init__(self,windowSize,dimensions):
        
        super().__init__()

        self.model = nn.Sequential(
                    torch.nn.Linear(dimensions*windowSize , 100),
                    torch.nn.ReLU(),
                    torch.nn.Linear(100 , 50),
                    torch.nn.ReLU(),
                    torch.nn.Linear(50 , 20),
                    torch.nn.ReLU(),
                    torch.nn.Linear(20 , 50),
                    torch.nn.ReLU(),
                    torch.nn.Linear(50 , 100),
                    torch.nn.ReLU(),    
                    torch.nn.Linear(100, dimensions*windowSize)
                )

        self.model.to(device)

    def forward(self,x):
        x = torch.transpose(x,0,1)
        x = self.model(x)
        return torch.transpose(x,0,1)


#model = RecurrentAutoencoder(sampleWindowSize, dimensions)
#model = BinCoder(dimensions,130,3)
model = ArnCoder(sampleWindowSize,dimensions)
model = model.to(device)


#####################################################
#           Training                                #
#####################################################

from torch import optim
import copy


optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
criterion = nn.L1Loss(reduction = "sum").to(device)
history = dict(train=[],val=[])

best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 1e9

for epoch in range(1,n_epochs +1):
    model = model.train()

    train_loss = []
    for seq_true in trainingSet:

        optimizer.zero_grad()
        
        seq_true = seq_true.to(device) # I think we can do this faster if we transfer the dataset to the gpu first and than do the training
        seq_pred = model(seq_true)
    
        loss = criterion(seq_pred, seq_true)

        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    val_loss = []

    model = model.eval()
    
    with torch.no_grad():
        for seq_true in validationSet:

            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred,seq_true)
            val_loss.append(loss.item())

    train_loss = np.mean(train_loss)
    val_loss = np.mean(val_loss)

    history["train"].append(train_loss)
    history["val"].append(val_loss)

    if val_loss < best_loss:
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())

    print(f"Epoch {epoch}: train loss {train_loss} val loss {val_loss}.")

model.load_state_dict(best_model_wts)


import matplotlib.pyplot as plt

plt.plot(history["train"])
plt.plot(history["val"])
plt.show()
plt.close()

seq_true1 = validationSet[0].to(device)
seq_pred1 = model(seq_true1)
seq_true2 = validationSet[1].to(device)
seq_pred2 = model(seq_true2)

plt.plot(validationSet[0])
plt.plot(seq_pred1.to("cpu").detach().numpy())
plt.show()
plt.close()
plt.plot(validationSet[1])
plt.plot(seq_pred2.to("cpu").detach().numpy())
plt.show()
