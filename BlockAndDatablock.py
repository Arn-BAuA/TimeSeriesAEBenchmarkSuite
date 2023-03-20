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

    def __init__(**hyperParameters):

        defaultHyperParameters = self._getDefaultHPs()

        HPs = {**defaultHyperParameters,**hyperParameters}
        self.HP = HPs




    #To be overwritten. Gets called in the 
    #__init__ of the children to get their default hp
    # in the constructor
    def _getDefaultHPs(self):
        pass

    def hyperParamteres(self):
        return self.HP


#The dataset wrapper itself is not an object. It is a file containing a method 
# that returns training set, validation set and testset.
# these set itself are datablock classes.
class DataBlock(block):

    def __init__(Dataset,**hyperParameters):
        block.__init__(hyperParameters)
        self.Dataset=Dataset

    def Data(self):
        return self.Dataset

    #Default HPs of DataBlocks are defined in the setwrapper
    def _getDefaultHPs(self):
        return {}
