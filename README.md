# Time Series Autoencoder Benchmark Suite

![](.mdpictures/Banner.png)

## Contents

## About this Repo and Basic Overview

This repository functions as a basis to evaluate the performance of various autoencodertypes for time series reproduction.
This work is part of a project which is aimed at time series anomaly detection, so there is a special focus on anomalies in
some parts of the code.
There are three major buildingblocks to create an autoencoder, datasets, model and training algorithm:

* The Data - The Data must be representative for the underlieng probabilitydistribution of the data source
and, more important here, there must be sufficient information in the dataset to match the information needed
to train a certain type of autoencoder (e.g. roughly: an LSTM autoencoder need more data than a feed forward Autoencoder
of similar size)
* The Model -  The Model must be suited to catch the information in the distribution that generates the data, while
beeing robust enough to withstand noise.
* The Training algorithm - The combination of batching strategy and optimizer and other steps for the training, that are
important to find the best weights for the model.

### Basic Architecture of Benchmark.py

![](.mdpictures/RepoOverview.png)

If these three buildingblocks are designed and tweaked the right way, good results will be archived.
The Codestructure of this repository is centered around these three blocks.

The Heart of the code is the Benchmark.py file. It is a wrapper, which coordinates the exchange of information between two
classes and one function that encapsulate these three blocks. The process encapsulated in the benchmark script is the following:

* The DatasetWrapper method encapsulates a dataset. As an argument, it just takes a wrapper specific dict of hyperparameters (Well come to that later) and a path for the output to the file system. It delivers three datasets: trainingset, validationset and testset. These are arrays of pytorch tensors of dimensionality d x n where d is the number of dimensions and n is the number of samples.
Characteristics that estimate the size and information in the sets are outputted to the file system at the specified path.
The files except a data folder that contains the raw versions of the wrapped dataset. THis folder is not included in the git repo (datasets are to large). TODO: Make an init.sh scipt that creates this folder.

* The Model is a class, that takes only a dict of hyperparameters in the constructor. It is a model that suits the autoencoder interface: Namely: It takes the input data and outputs the reconstruction.

* The Trainier is a method that takes a training and a validation set, a model, a dictionary of hyperparameters and a path for outputs to the file system. It than trains the model on training set and validates it with the testset. The trainer methode gathers a bunch of performace characteristics, example outputs and model weights at diffrent points of the training. They will be written to the filesystem in a specified location. The Trainermethod outputs the trained model.

The trained model will than be tested on the testset. Performance characteristics and weights of the trained model are saved to the file system.
The diffrent methods that implement the interfaces specified above are stored in the folders in the repo in separate python files. Often used hyperparameter combinations are stored along with them in a folder that must be named like the script containing the module itself.
The Benchmark method takes the used classes methods , paths and hyperparameterdicts as inputs

### Block and Datablock

The Hyperparameter storage and handling doesnt change between the blocks that are mentioned above. The block and datablock handle the hyperparameter management on an abstract polymorph level. Trainer and Model inhere from block. The situation with the dataset is a bit more complicated, since the setwrapper itself is not an object. Its just a method, that return the datasets, which are of class DataBlock. A generic implementation of Block for all Datasets.

### Working with the Suite

The benchmark scipt has to be called form somewhere. We dont call it directly. We use an experiment script.
The experiment script takes the buildingblocks defined above and does an experiment. Individual experiments are stored in the experiments folder but are ment to be executed in the root of the repo.
In the Benchmark.py file are a bunch of methods to create the nessecairy dicts and save the hyperparameters and environment data.
Here a small example scipt, benchmarking a feed forward autoencoder on syntehtic sine data and visualizing the output:
<pre><code>
#!/bin/python

from Models.FeedForward import Model as FeedForwardAE
from DataGenerators.Sines import generateData as Sines
from Trainers.SingleInstanceTrainer import Trainer as OnlineTrainer

from Benchmark import benchmark,initializeDevice
from Evaluation.QuickOverview import plotOverview

pathToSave = "Tech Demo"

device = initializeDevice()
Dimensions = 2 # Dataset dimensions


trainingSet,validationSet,testSet = Sines(Dimensions)

model = FeedForwardAE(Dimensions,device)

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

</code></pre>

The scripts in the Evaluation dict are used to evaluate the data on the fs generated by the experiment. In evaluation is a subdirectory that is called Utility\_Plot which contains templates for often used plots and an directory called Utility\_Data which encapsulates some helpers for loading and saving data.





## The Benchmark.py file

## The Experiment.py file

### SetWrappers/\*.py

### AEModels/\*.py

### Trainers/\*.py

### Experiments/\*.py

### Evaluation/\*.py

## More Example Code
