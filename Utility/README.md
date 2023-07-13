
![](../.mdpictures/Banners/Utility.png)

This folder houses utility methods that are potentially useful in multiple areas of the script (Thus they are not in a utility module local in some other folder in the repository).
Below is a list of the modules and a brief description of their methods.<br>

## DataSetSampler.py

* RandomSampling: This method is used in the SMD-setwrapper, the Airquality-setwrapper and the Sines data generator. It randomly samples snippets of a given window size from a large time series.
* fromClassification2AnoDetect: This method is used for the creation of the UCR and ECG dataset in the coresponding set wrappers. It is used to transform a classification data set to an anomaly detection data set, by random sampling from the classes and labling one type anomal. Pleas read the description of the UCR and ECG setwrappers for more information.
* splitDataframeAlongRows: This method cuts a pandas dataframe into smaller dataframes by cutting "vertically" trough the rows. It is used in the SMD-setwrapper.
* selectByLabel: A little helper function, that returns a copy of a passed data frame that contains only selected labels.

## MathOperations.py

Often used operations, that are not implemented in pandas, torch or numpy are stored here. At the moment, there is only a function normalizing a dataframe.

