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
            "Adiac",
            "AllGestureWiimoteY",
            "AllGestureWiimoteZ",
            "ArrowHead",
            "Beef",
            "BeetleFly",
            "BirdChicken",
            "Car",
            "CBF",
            "ChlorineConcentration",
            "CinCECGTorso",
            "Coffee",
            "Computers",
            "CricketX",
            "CricketY",
            "CricketZ",
            "DiatomSizeReduction",
            "DistalPhalanxOutlineAgeGroup",
            "DistalPhalanxOutlineCorrect",
            "DistalPhalanxTW",
            "DodgerLoopGame",
            "DodgerLoopWeekend",
            "Earthquakes",
            "ECG200",
            "ECG5000",
            "ECGFiveDays",
            "ElectricDevices",
            "EOGVerticalSignal",
            "FaceAll",
            "FaceFour",
            "FacesUCR",
            "FiftyWords",
            "Fish",
            "FordA",
            "FordB",
            "FreezerSmallTrain",
            "GestureMidAirD1",
            "GestureMidAirD2",
            "GesturePebbleZ1",
            "GunPointAgeSpan",
            "GunPointMaleVersusFemale",
            "GunPointOldVersusYoung",
            "Ham",
            "HandOutlines",
            "Haptics",
            "Herring",
            "InlineSkate",
            "InsectEPGRegularTrain",
            "InsectEPGSmallTrain",
            "InsectWingbeatSound",
            "ItalyPowerDemand",
            "LargeKitchenAppliances",
            "Lightning2",
            "Lightning7",
            "Mallat",
            "Meat",
            "MedicalImages",
            "MelbournePedestrian",
            "MiddlePhalanxOutlineAgeGroup",
            "MiddlePhalanxOutlineCorrect",
            "MiddlePhalanxTW",
            "MixedShapesSmallTrain",
            "MoteStrain",
            "NonInvasiveFetalECGThorax1",
            "NonInvasiveFetalECGThorax2",
            "OliveOil",
            "OSULeaf",
            "PhalangesOutlinesCorrect",
            "Phoneme",
            "PickupGestureWiimoteZ",
            "PigArtPressure",
            "PigCVP",
            "Plane",
            "ProximalPhalanxOutlineAgeGroup",
            "ProximalPhalanxOutlineCorrect",
            "ProximalPhalanxTW",
            "RefrigerationDevices",
            "ScreenType",
            "ShakeGestureWiimoteZ",
            "ShapeletSim",
            "ShapesAll",
            "SmallKitchenAppliances",
            "SonyAIBORobotSurface1",
            "SonyAIBORobotSurface2",
            "StarLightCurves",
            "Strawberry",
            "SwedishLeaf",
            "Symbols",
            "SyntheticControl",
            "ToeSegmentation1",
            "ToeSegmentation2",
            "Trace",
            "TwoLeadECG",
            "TwoPatterns",
            "UWaveGestureLibraryAll",
            "UWaveGestureLibraryX",
            "UWaveGestureLibraryY",
            "UWaveGestureLibraryZ",
            "Wine",
            "WordSynonyms",
            "Worms",
            "WormsTwoClass",
            "Yoga",
    ]


def getNumDS():
    return 173

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
    
    if DataSrcNumber < 128:
        return UCRDataSet(Dimensions,DataSet = selectedUCRSets[DataSrcNumber-7],**hyperParameters)
    
    SMDMachines = [[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],[2,1],[2,2],[2,3],[2,4],[2,5],[2,6],[2,7],[2,8],[2,9],[3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],[3,8],[3,9],[3,10],[3,11]]
    
    if DataSrcNumber < 156:
        return SMDDataSet(Dimensions,nNormalDimensions=0,machineType = SMDMachines[DataSrcNumber-129][0],machineIndex = SMDMachines[DataSrcNumber-129][1],**hyperParameters)
    
    
    if DataSrcNumber == 156:
        return Sines(Dimensions,AnomalousAmplitudes=[[2.2],[1.2]],**hyperParameters)
    if DataSrcNumber == 157:
        return Sines(Dimensions,AnomalousFrequency=[[0.5],[2.2]],**hyperParameters)
    if DataSrcNumber == 158:
        return Sines(Dimensions,NoiseLevel = 0.2,**hyperParameters)
    if DataSrcNumber == 159:
        return Sines(Dimensions,AnomalousAmplitudes=[[6.2],[6.2]],**hyperParameters)
    if DataSrcNumber == 160:
        return Sines(Dimensions,AnomalousFrequency=[[6],[6.2]],**hyperParameters)
    if DataSrcNumber == 161:
        return Sines(Dimensions,NoiseLevel = 0.4,**hyperParameters)
    if DataSrcNumber == 162:
        return Sines(Dimensions,AnomalousAmplitudes=[[0.1],[0.1]],**hyperParameters)
    if DataSrcNumber == 163:
        return Sines(Dimensions,AnomalousFrequency=[[2],[2]],**hyperParameters)
    if DataSrcNumber == 164:
        return Sines(Dimensions,NoiseLevel = 2,**hyperParameters)
    if DataSrcNumber == 165:
        return Sines(Dimensions,AnomalousAmplitudes=[[3],[0.2]],**hyperParameters)
    if DataSrcNumber == 166:
        return Sines(Dimensions,AnomalousFrequency=[[3],[5]],**hyperParameters)
    if DataSrcNumber == 167:
        return Sines(Dimensions,NoiseLevel = 0.7,**hyperParameters)
    if DataSrcNumber == 168:
        return Sines(Dimensions,AnomalousAmplitudes=[[0.5],[0.5]],**hyperParameters)
    if DataSrcNumber == 169:
        return Sines(Dimensions,AnomalousFrequency=[[0.02],[0.2]],**hyperParameters)
    if DataSrcNumber == 170:
        return Sines(Dimensions,NoiseLevel = 1,**hyperParameters)
    if DataSrcNumber == 171:
        return ECGDataSet(Dimensions,**hyperParameters,ArrythmiaNormals = [0],ArrythmiaAnomalys=[1,2,3,4],PTBNormals=[0],PTBAnomalys=[1])
    if DataSrcNumber == 172:
        return ECGDataSet(Dimensions,**hyperParameters,ArrythmiaNormals = [],ArrythmiaAnomalys=[],PTBNormals=[1],PTBAnomalys=[0])
    if DataSrcNumber == 173:
        return ECGDataSet(Dimensions,**hyperParameters,ArrythmiaNormals = [],ArrythmiaAnomalys=[],PTBNormals=[0],PTBAnomalys=[1])
