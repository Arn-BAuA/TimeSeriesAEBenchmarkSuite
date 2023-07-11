
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


<pre><co
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
de>
</pre></code>
