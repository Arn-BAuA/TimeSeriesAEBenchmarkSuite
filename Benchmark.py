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
n_epochs = 300 #for Training

#################################
# Data Loading                  #
#################################

relevantColumns = Columns[0:dimensions]

#loading and preparing air quality data
def loadAirQualityData():
    dataset = pd.read_excel(PathToAirqualityData,parse_dates=[["Date","Time"]])
    dataset = dataset[["Date_Time"]+relevantColumns]
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
    
    return DataSet

trainingSet = SampleDataSet(datetime(2004,4,1),datetime(2005,1,1),100)
validationSet = SampleDataSet(datetime(2005,1,1),datetime(2005,3,1),50)
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


# First Example, inspired by a Tutorial on Curiously.com by 


class Encoder(nn.Module):

    def __init__(self,seq_len, input_dim, latent_dim = 64):

        super(Encoder,self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.inner_dim, self.outer_dim = latent_dim, 2*latent_dim

        self.outerRNN = nn.LSTM(
                    input_size = input_dim,
                    hidden_size = self.outer_dim,
                    num_layers = 1,
                    batch_first = True
                )

        self.innerRNN = nn.LSTM(
                    input_size = self.outer_dim, #warum? müsste es nicht auch input_dim sein?
                    hidden_size = self.inner_dim,
                    num_layers =  1,
                    batch_first = True
                )

    def forward(self,x):
        x= x.reshape((1,self.seq_len,self.input_dim))

        x,(_, _) = self.outerRNN(x)#?? Wieso gebe ich hier x weiter und nicht den inneren zustand?
        x, (hidden_n,_) = self.innerRNN(x)#??

        return hidden_n.reshape((self.input_dim,self.inner_dim))

class Decoder(nn.Module):

    def __init__(self, seq_len, input_dim, latent_dim = 64):

        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.inner_dim, self.outer_dim = latent_dim, 2*latent_dim

        self.innerRNN = nn.LSTM(
                input_size=latent_dim,
                hidden_size = latent_dim,
                num_layers = 1,
                batch_first = True
                )

        self.outerRNN = nn.LSTM(
                input_size = latent_dim,
                hidden_size = self.outer_dim,
                num_layers = 1,
                batch_first = True
                )

        self.output_layer = nn.Linear(self.outer_dim, input_dim)

    def forward(self,x):
        x = x.repeat(self.seq_len,self.input_dim)
        x = x.reshape((self.input_dim, self.seq_len, self.inner_dim))

        x, (hidden_n,cell_n) = self.innerRNN(x)
        x, (hidden_n,cell_n) = self.outerRNN(x)
        x = x.reshape(self.seq_len, self.outer_dim)#??

        return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):

    def __init__(self, seq_len, input_dim, latent_dim = 64):
        
        super(RecurrentAutoencoder,self).__init__()

        self.encoder = Encoder(seq_len, input_dim, latent_dim).to(device)
        self.decoder = Decoder(seq_len, input_dim, latent_dim).to(device)

    def forward(self , x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = RecurrentAutoencoder(sampleWindowSize, dimensions)
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

seq_true = validationSet[1].to(device)
seq_pred = model(seq_true)

plt.plot(validationSet[1])
plt.plot(seq_pred.to("cpu").detach().numpy())
plt.show()
