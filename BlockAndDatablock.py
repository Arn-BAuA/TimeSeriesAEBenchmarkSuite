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

    def __init__(self,**hyperParameters):

        defaultHyperParameters = self._getDefaultHPs()

        HPs = {**defaultHyperParameters,**hyperParameters}
        self.HP = HPs




    #To be overwritten. Gets called in the 
    #__init__ of the children to get their default hp
    # in the constructor
    def _getDefaultHPs(self):
        return {}

    def hyperParamteres(self):
        return self.HP


#The dataset wrapper itself is not an object. It is a file containing a method 
# that returns training set, validation set and testset.
# these set itself are datablock classes.
class DataBlock(block):

    def __init__(self,Dataset,Dimensions,**hyperParameters):
        block.__init__(self,**hyperParameters)
        self.Dataset=Dataset
        self.Dimesions = Dimensions

    def Data(self):
        return self.Dataset

    def Dimension(self):
        return self.Dimensions

