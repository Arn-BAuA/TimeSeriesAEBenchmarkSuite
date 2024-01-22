

import matplotlib.pyplot as plt
from Evaluation.Utility_Plot.ErrorPlotter import plotErrors,plotErrorsAgainstExample 
from Evaluation.Utility_Plot.MilestonePlotter import plotMilestones
from Evaluation.Utility_Plot.ExamplePlotter import plotExample
from Evaluation.Utility_Plot.HistogramAndExample import plotExampleLocation
from Evaluation.Utility_Plot.MilestoneHistograms import plotMilestoneHistograms
import os
import glob
import json
from Evaluation.HTMLSummary import createHTMLSummary


#################################
#
#



def plotOverview(rootDir):
    
    overviewDir = rootDir+"Overview/"
    pngDir = overviewDir+"PNG/"
    pdfDir = overviewDir+"PDF/"

    os.mkdir(overviewDir)
    os.mkdir(pngDir)
    os.mkdir(pdfDir)
    
    def createPlot(PNGpath,PDFpath,method,r_d,**methodArgs):
        fig, ax = plt.subplots()

        method(r_d,ax,**methodArgs)

        plt.savefig(PNGpath+".png")
        plt.savefig(PDFpath+".pdf")
        plt.close()
   
    metaDataFile = open(rootDir+"HyperParametersAndMetadata.json","r")
    runMetadata = json.load(metaDataFile)
    metaDataFile.close()

    for errorName in runMetadata["Used Errors"]: 
        createPlot(pngDir+errorName+" against Epochs",pdfDir+errorName+" against Epochs",plotErrors,rootDir,errorName = errorName,against = "Epoch")
        createPlot(pngDir+errorName+" against CPUTime",pdfDir+errorName+" against CPUTime",plotErrors,rootDir,errorName = errorName,against = "CPUTime")
        createPlot(pngDir+errorName+" against Wall Time",pdfDir+errorName+" against Wall Time",plotErrors,rootDir,errorName = errorName,against= "WallTime")

        createPlot(pngDir+errorName+" Histograms on Training Set",pdfDir+errorName+" Histograms on Training Set",plotMilestoneHistograms,rootDir,errorName=errorName,setName="Training Set")
        createPlot(pngDir+errorName+" Histograms on Validation Set",pdfDir+errorName+" Histograms on Validation Set",plotMilestoneHistograms,rootDir,errorName=errorName,setName="Validation Set")
        createPlot(pngDir+errorName+" Histograms on Test Set",pdfDir+errorName+" Histograms on Test Set",plotMilestoneHistograms,rootDir,errorName=errorName,setName="Test Set")

    for file in glob.glob(rootDir+"Final Model/*.csv"):
        file = file.split("/")[-1]
        createPlot(pngDir+"Milestones for "+file,pdfDir+"Milestones for "+file,plotMilestones,rootDir,ExampleName=file)
        createPlot(pngDir+"Data in "+file,pdfDir+"Data in "+file,plotExample,rootDir,ExampleName=file)
        
        setName = file.split(" ")[0]
        exampleIndex = int(file.split("(")[-1][:-5]) #splitting of ").csv hence -5"
        errorName = runMetadata["Used Errors"][0]
        
        createPlot(pngDir+" Error Comparison for "+file,pdfDir+" Error Comparison for "+file,plotErrorsAgainstExample,rootDir,errorName=errorName,setName = setName,exampleIndex = exampleIndex)
        createPlot(pngDir+" Error Location for "+file,pdfDir+" Error Location for "+file,plotExampleLocation,rootDir,errorName=errorName,setName = setName+" Set",exampleIndex = exampleIndex)
    createHTMLSummary(rootDir)

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        raise ValueError("Please Specify the path to the directory of the experiment.")

    rootDir = sys.argv[1]

    plotOverview(rootDir)
