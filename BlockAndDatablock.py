####
##
##
# The Benchmark routine in this script takes diffrent blocks.
# The model, the dataset, and the trainier
# Thises buildingblock all have something in comon
# they should contain the main content (model, dataset, trainer)
# as well as the hyperparameters that where chosen to create said content, since
# content and hyperparameters are mostly used together.
# THe block and datablock classes handle the interface and hyperparameter management

# abstract class for trainer and model to handle the hyperparameters
class block():
    
    #name is the name of the specific dataset, model or trainer for the documentation later.
    def __init__(self,name,**hyperParameters):
        
        defaultHyperParameters = self._getDefaultHPs()

        HPs = {**defaultHyperParameters,**hyperParameters}
        self.HP = HPs
        
        self.name = name



    #To be overwritten. Gets called in the 
    #__init__ of the children to get their default hp
    # in the constructor
    def _getDefaultHPs(self):
        return {}

    def Name(self):
        return self.name

    def hyperParameters(self):
        return self.HP


#The dataset wrapper itself is not an object. It is a file containing a method 
# that returns training set, validation set and testset.
# these set itself are datablock classes.
class DataBlock(block):

    def __init__(self,name,Dataset,Dimensions,**hyperParameters):
        block.__init__(self,name,**hyperParameters)
        self.Dataset=Dataset
        self.Dimensions = Dimensions
        self.haslabels = False

    #if you have labeled data or you are using a data generator, you can add
    #the gound truth here. It should be an array of the same size as the set,
    # containing an array of 0 and 1 where 0 means that its no anomaly and one
    # means it is.
    def setLabels(self,isAnomaly):
        self.labels= isAnomaly
        self.haslabels = True

    def hasLabels(self):
        return self.haslabels

    def Labels(self):
        return self.labels

    def Data(self):
        return self.Dataset

    def Dimension(self):
        return self.Dimensions

    def Length(self):
        return self.Dataset[0].shape[0]

