
import os
dirname = os.path.dirname(__file__)
SMDPath = os.path.join(dirname,"../data/SMD/")

def loadData(dimensions,**hyperParameters)
    
    defaultHyperParameters = {
            "machineType":1,#Machine type of the smd dataset (number between 0 and 1)
            "machineIndex":1,#the machine index from the SMD Dataset
            "nNormalDimensions":0,#In the SMD Dataset are dimensions which don't contribute to any anomaly. these are the normal dimnsions. This algorithm selects the dimensions by anomaly. the most anomal one getts added fists, than the second and so forth. Except we demand normal dimensions by using this hyperparameter. If it is e.g. 3 we demand that the 3 least anomal dimensions are used as well.
            "ValidationSetContainsAnomalys":True,
            "ValidationSetSplit":50,# The percentage of the set where the validation set originates that is split off for the validation.
            "TrainingSetSize":400,
            "ValidationSetSize":100,
            "TestSetSize":30,
            "SampleLength":150,# The length of the snippets that are generated.
        }

    HPs={**defaultHyperParameters,**hyperParameters}
