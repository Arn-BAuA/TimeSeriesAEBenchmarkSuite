#!/bin/python

import os
import json
import glob

def thumbnailImage(imgName):
    src = "Overview/PNG/"+imgName
    pdfSrc = "Overview/PDF/"+imgName[:-4]+".pdf"

    return "<a href=\""+pdfSrc+"\"><img src=\""+src+"\" class =\"thumbnailImg\"></a>\n"

def biggerThumbnailImage(imgName):
    src = "Overview/PNG/"+imgName
    pdfSrc = "Overview/PDF/"+imgName[:-4]+".pdf"

    return "<a href=\""+pdfSrc+"\"><img src=\""+src+"\" class =\"bigThumbnailImg\"></a>\n"

def addToHTML(html,placeholder,content):
    split = html.split(placeholder)
    return split[0] + content + split[1]

def createHTMLSummary(rootDir):

    if rootDir[-1] == '/':
        rootDir = rootDir[:-1]

    templateFile = open("Evaluation/HTMLRescources/OverviewTemplate.html",'r')
    htmlDoc ="" 
    for line in templateFile:
        htmlDoc+=line
    templateFile.close()

    htmlDoc = addToHTML(htmlDoc,"HeadlineHere",rootDir.split('/')[-1])

    ##############################
    ##
    # Loading and adding Hyperparameters:
    
    HPFile = open(rootDir+"/HyperParametersAndMetadata.json",'r')
    HPs = json.load(HPFile)
    HPFile.close()
    
    def formatDict(dictionary):
        params = json.dumps(dictionary,indent=4)
        params = params.replace(" ","&nbsp;")
        params = params.replace("\n","<br>")
        return params

    htmlDoc = addToHTML(htmlDoc,"DataSourceHere",HPs["General Information"]["Used Dataset"])
    htmlDoc = addToHTML(htmlDoc,"DataHPsHere",formatDict(HPs["SetWrapper HPs"]))

    htmlDoc = addToHTML(htmlDoc,"ModelTypeHere",HPs["General Information"]["Used Model"])
    htmlDoc = addToHTML(htmlDoc,"ModelHPsHere",formatDict(HPs["Model HPs"]))

    htmlDoc = addToHTML(htmlDoc,"TrainerTypeHere",HPs["General Information"]["Used Trainer"])
    htmlDoc = addToHTML(htmlDoc,"TrainerHPsHere",formatDict(HPs["Trainer HPs"]))
    
    MainInfo = "Repo V.: "+HPs["GITHash"]
    MainInfo+= "<br>#Dimemsions: "+str(HPs["General Information"]["StreamDimension"])
    MainInfo+= "<br>#Epochs: "+str(HPs["General Information"]["Number of epochs"])
    MainInfo+= "<br>Computation Device: "+HPs["Hardware Info"]["UsedComputationDevice"]

    htmlDoc = addToHTML(htmlDoc,"GitHeadHashHere",MainInfo)

    #############################
    ##
    # Loading and adding Errors
    
    errorPortion = "" #Stuff that gets added to html

    for Error in HPs["Used Errors"]:
        errorPortion += "<h3> "+Error+"</h3>\n"
        
        againstEpochs = ""
        againstCPUTime = ""
        againstWallTime = ""

        for img in glob.glob(rootDir+"/Overview/PNG/*"+Error+"*"):
            fileName = img.split('/')[-1]

            if "Epoch" in fileName or "epoch" in fileName:
                againstEpochs = biggerThumbnailImage(fileName)
                continue
            if "CPU" in fileName:
                againstCPUTime = biggerThumbnailImage(fileName)
                continue
            if "Wall" in fileName:
                againstWallTime = biggerThumbnailImage(fileName)
                continue
            if "Histogram" in fileName:
                if "Training" in fileName:
                    histogramTS = biggerThumbnailImage(fileName)
                    continue
                if "Validation" in fileName:
                    histogramVS = biggerThumbnailImage(fileName)
                    continue
                if "Test" in fileName:
                    histogramTest = biggerThumbnailImage(fileName)
                    continue
        
        errorPortion += againstEpochs+againstWallTime+againstCPUTime
        errorPortion += histogramTS+histogramVS+histogramTest

    htmlDoc = addToHTML(htmlDoc,"PerformanceCharacteristicsHere",errorPortion)

    #############################
    ##
    # Loading and adding Examples
    
    examplePortion = ""   
    
    for ExampleType in HPs["Selected Examples"]:
        
        examplePortion+= "<h3> " +ExampleType+ " </h3>"
        
        indices = HPs["Selected Examples"][ExampleType][1:-1].split(' ')
        

        for index in indices:
            
            if len(index) == 0:
                continue

            index = int(index)

            availablePlots = glob.glob(rootDir+"/Overview/PNG/*"+ExampleType+"("+str(index)+")"+"*")

            otherFiles = []

            for path in availablePlots:
                name = path.split('/')[-1]

                if "Data" in name:
                    dataFile = thumbnailImage(name)
                    continue
                if "Milestone" in name:
                    milestoneFile = thumbnailImage(name)
                    continue
                if "Location" in name:
                    locationFile = thumbnailImage(name)
                    continue
                if "Comparison" in name:
                    compareFile = thumbnailImage(name)
                    continue
                otherFiles.append(thumbnailImage(name))
            
            examplePortion += dataFile+milestoneFile+compareFile+locationFile
            for file in otherFiles:
                examplePortion+=file
            
    htmlDoc = addToHTML(htmlDoc,"ExamplesHere",examplePortion)

    ############################
    ###
    # Write Files

    outFile = open(rootDir+"/Summary.html",'w')
    outFile.write(htmlDoc)
    outFile.close()

if __name__ == "__main__":
    
    import sys
    path = sys.argv[1]
    createHTMLSummary(path)
