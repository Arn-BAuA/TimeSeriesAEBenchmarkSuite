
![](../.mdpictures/Banners/SetWrappersBanner.png)

SetWrappers are the way data is loaded into the framework. A setwrapper is a function that loads a data set from the raw data that is available on the internet, or is measuret in some experiment and outputs three datasets (object of type datablock). The three datasets are Trainingset, ValidationSet and Test set.<br>
In other words: a set wrapper takes the raw uncleaned data and outputs data that behaves like it is expected by the other interfaces in the framework. The Hyperparameters that are passed to the diffrent setwrappers depend on the data that is wrapped. <br>
In the following passages, each wrapper and its hyperparameters are discussed:

## ECGDataSet:



The ECG Heartbeat categorisation set (https://www.kaggle.com/datasets/shayanfazeli/heartbeat) is a set that is based on two other sets. The MIT-BIH Arythmia Dataset and the PTB Diagnositc ECG Database. The first of them features 5 classes. One normal class and four diffrent kinds of anomalies. The latter one only has two kinds of labels, normal data and abnormal data.<br>

### Short Descirption
The data is made out of time series snippets which have the same size (they are zero padded to have the same size). The Set wrapper basically creates the training-, validation- and test-set by sampling random snippets. <br>
When multiple dimensions are demanded, multiple one dimensional snippets are stiched together to create a multivariate time series. <br>
Anomalies are included at random at a rate the user can choose individually for each of the three output sets. Anomalies can either occur in all dimensions or just in a fraction of the dimensions.

### Hyperparameters 

The hyperparameters are documented in the code:

<pre><code>
    defaultHyperParameters = {
            "ArrythmiaNormals":[0],#Classes from arrythmia that are used as normals, none, if left emty
            "ArrythmiaAnomalys":[1,2,3,4],#classes from arrythmia that are used as anomalies, none if left empty
            "PTBNormals":[],#Classes from PTB that are concidered as normal, none if left empty
            "PTBAnomalys":[],#Classes from the PTB that are used as anomalies, none if left empty
            "AnomalyPercentageTrain":0,#Percentage of anomalies in training set
            "AnomalyPercentageValidation":10, #percentage of anomalies in validation set
            "AnomalyPercentageTest":10,# percentage of anomalies in test set
            "SameClassForAllDimensions":False,#when true: All dimensions are of the same class, else, they are random
            "AllDimensionsAnomal":False,#If this is true and the value is an anomaly, all the dimensions are an anomaly
            "nAnomalDimensions":1,#if it is an anomaly: How many dimensions are anomal
            "UseArythmiaSplit":True,#For the used arythmia data: Use the split for training and test like in the original csv files:
            "PercentTrainingSet":70,# The percentage of samples that is split off for the training.
            "PercentValidationSet":20, # The percentage of samples that is split off for the validation.
            "PercentTestSet":10, # The percentage of samples that is split off for th testing.
            "TrainingSetSize":400,
            "ValidationSetSize":100,
            "TestSetSize":30,
        }
</pre></code>

## UCRArchive

This is a setwrapper for the 2018 time series classification archive (https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).<br>
The UCR-Archive is a collection of different time series data sets from different domains. The Aim here is to create a collection of data sets for benchmarking that covers most of the things that can be encountered in the field.<br>
There is also a UCR anomaly detection archive, which we are not supporting at the moment.<br>
The Archive consist of multiple univariate time series datasets with class labels.

### Short Description

This data set wrapper has much in common with the wrapper for the ECG-Set. It creates the three sets for the framework by randomly sampling time series snippets from the archive.<br>
It can also output multivariate data by stitching univariate examples on top of each other.<br>
Anomalies are added at random at a rate, the user can set individually for the three sets. Anomalies either occur in some or all dimensions of a given example of the time series.

### Hyperparameters:

The hyperparameters are documented in the code:

<pre><code>
defaultHyperParameters = {
            "DataSet":"UMD",#Name of the dataset
            "AnomalyClass":3,#Class that is picked as anomal
            "AnomalyPercentageTrain":0,#Percentage of anomalies in training set
            "AnomalyPercentageValidation":10, #percentage of anomalies in validation set
            "AnomalyPercentageTest":10,# percentage of anomalies in test set
            "SameClassForAllDimensions":False,#when true: All dimensions are of the same class, else, they are random
            "AllDimensionsAnomal":False,#If this is true and the value is an anomaly, all the dimensions are an anomaly
            "nAnomalDimensions":1,#if it is an anomaly: How many dimensions are anomal
            "SmallestClassAsAnomaly":True, #if true, the entrie of AnomalyClass is overwritten and the smalles class is taken as anomal.
            "KeepTrainAndTestStructure":False,#if set true, the samples for training and validation are drawn from the TRAIN and TEST file in the UCR Archive. If set false, they will be mixed.
            "TrainingSetSize":400,
            "ValidationSetSize":100,
            "TestSetSize":30,
        }


</pre></code>

## SMD-Setwrapper

The SMD (Server Machine Dataset) is a set containing multivariate telemetry data from different servers. Anomalies are labeled. The set has been published along side the OmnyAnomaly-algorithm (https://github.com/NetManAIOps/OmniAnomaly).<br>

### Short Description:

The SMD contians telemetry data from roughly 30 machines. The data is stored in long time series. Anomalies only manifest them selves in a portion of the dimensions of the time series. The timestamps and the dimension at which anomalies occur are labeled.<br>
The Wrapper cuts small snippets out of the huge time series for one machine and, if demanded, reduces the dimensions to a number of dimensions specified by the user.<br>
When reducing dimensions, the algorithm sorts the dimensions according to the number of anomalies that manifest in them. The once with the most anomalies are kept. In addition, the user can specify to include a number of dimensions with low anomaly count, leading essentially to the possibility to specify how many of the reduced dimensions should be normal and how many should contain anomalies.<br>
The user can also specify the rate at which anomalies occur for each individual dimension.

### Hyperparameters


<pre><code>
defaultHyperParameters = {
            "machineType":1,#Machine type of the smd dataset (number between 0 and 1)
            "machineIndex":1,#the machine index from the SMD Dataset
            "nNormalDimensions":0,#In the SMD Dataset are dimensions which don't contribute to any anomaly. these are the normal dimnsions. This algorithm selects the dimensions by anomaly. the most anomal one getts added fists, than the second and so forth. Except we demand normal dimensions by using this hyperparameter. If it is e.g. 3 we demand that the 3 least anomal dimensions are used as well.
            "ValidationSetContainsAnomalys":True,
            "ValidationSetSplit":50,# The percentage of the set where the validation set originates that is split off for the validation.
            "NormalizeValues":True,
            "TrainingSetSize":400,
            "ValidationSetSize":100,
            "TestSetSize":30,
            "SampleLength":150,# The length of the snippets that are generated.
        }


</pre></code>

## AirQualityUCI

The air quality UCI data set (https://www.kaggle.com/datasets/aayushkandpal/air-quality-time-series-data-uci) contains air quality measurements that where taken in a city in italia between 2004 and 2005.<br>
The data is unlabeled and contains some passages where data is missing.

### Short Description

The data set basically contains one large time series that is the complete measurement. The algorithm cuts randomly sampled snippets out of the time sereis.<br>
The user can specify which portions of the time series should be sampled for each of the three output sets.<br>
This wrapper is currently not up to date with the rest of the library.

### Hyperparameters

The hyperparameters are documented in the code.

<pre><code>
    defaultHyperParameters = {
            "sampleWindowSize" : 150,
            "includeTimeStamps" :False,
            #Due to the missing values in the DF which are not cleand, we create the oportunity to
            # chose manually, which intervals to use for which set. THe Dates are chosen in such a way,
            # that there are more holes in the test set than in the training set.
            "TrainingSetSize" : 1000,
            "BeginDateTrainingData" : datetime(2004,4,1),
            "EndDateTrainingData" : datetime(2005,1,1),
            "ValidationSetSize" : 100,
            "BeginDateValidationData" : datetime(2005,1,1),
            "EndDateValidationData" : datetime(2005,3,1),
            "TestSetSize" : 30,
            "BeginDateTestData" : datetime(2005,3,1),
            "EndDateTestData" : datetime(2005,4,1),
            }
 
</pre></code>
