

import matplotlib.pyplot as plt
from Evaluation.Utility_Plot.ErrorPlotter import plotErrors 
from Evaluation.Utility_Plot.MilestonePlotter import plotMilestones
from Evaluation.Utility_Plot.ExamplePlotter import plotExample
import os
import glob

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

    createPlot(overviewDir+"Errors",overviewDir+"Errors",plotErrors,rootDir)

    for file in glob.glob(rootDir+"Final Model/*.csv"):
        file = file.split("/")[-1]
        createPlot(pngDir+"Milestones for "+file,pdfDir+"Milestones for "+file,plotMilestones,rootDir,ExampleName=file)
        createPlot(pngDir+"Data in "+file,pdfDir+"Data in "+file,plotExample,rootDir,ExampleName=file)
    

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        raise ValueError("Please Specify the path to the directory of the experiment.")

    rootDir = sys.argv[1]

    plotOverview(rootDir)
