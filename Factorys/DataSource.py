# THe data source factory module. With all the methods to easylie access diffrent datasets.

from SetWrappers.UCRArchive import loadData as UCRDataSet
from SetWrappers.SMD import loadData as SMDDataSet
from SetWrappers.ECGDataSet import loadData as ECGDataSet
from DataGenerators.Sines import generateData as Sines

selectedUCRSets = [
            "ACSF1",
            "AllGestureWiimoteX",
            "BME",
            "Chinatown",
            "Crop",
            "DodgerLoopDay",
            "EOGHorizontalSignal",
            "EthanolLevel",
            "FreezerRegularTrain",
            "Fungi",
            "GestureMidAirD3",
            "GesturePebbleZ2",
            "GunPoint",
            "HouseTwenty",
            "MixedShapesRegularTrain",
            "PigAirwayPressure",
            "PLAID",
            "PowerCons",
            "Rock",
            "SemgHandGenderCh2",
            "SemgHandMovementCh2",
            "SemgHandSubjectCh2",
            "SmoothSubspace",
            "UMD",
            "Wafer",
    ]


numDataSrcs = 32

def getStandardDataSource(Dimensions,DataSrcNumber,**hyperParameters):

    if DataSrcNumber == 0:
        return Sines(Dimensions,**hyperParameters)
    if DataSrcNumber == 1:
        return Sines(Dimensions,AnomalousAmplitudes=[[1.2],[1.2]],**hyperParameters)
    if DataSrcNumber == 2:
        return Sines(Dimensions,AnomalousFrequency=[[1],[1.2]],**hyperParameters)
    if DataSrcNumber == 3:
        return Sines(Dimensions,NoiseLevel = 0.1,**hyperParameters)
    

    if DataSrcNumber == 4:
        return ECGDataSet(Dimensions,**hyperParameters)

    if DataSrcNumber == 5:
        return SMDDataSet(Dimensions,nNormalDimensions=0,**hyperParameters)
    if DataSrcNumber == 6:
        return SMDDataSet(Dimensions,nNormalDimensions=int(Dimensions*0.5),**hyperParameters)
    
    return UCRDataSet(Dimensions,DataSet = selectedUCRSets[DataSrcNumber-1],**hyperParameters)


