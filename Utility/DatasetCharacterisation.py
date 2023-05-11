import scipy.stats.differential_entropy as entorpy
import sklearn.metrics.mutual_info_score as mutualInformaion

#Small helper function to extract all normal/ anomal 
#Values from a Dataset
def aggregate(DataSet,AggregateNormal = True):

    Data = DataSet.Data()
    Anomaly = DataSet.Labels()

def calculateNormalEntropy(DataSet):

def calculateMutualInformationNormalAnomal(DataSet):
    

def score(DataSet):
    E_Score = calculateNormalEntropy(DataSet)
    MI_Score = calculateMutualInformationNormalAnomal(DataSet)
    return E_Score,MI_Score


