import glob
import sys
import matplotlib.pyplot as plt
import pandas as pd

def plotExample(rootDir,ax,ExampleName):

    data = pd.read_csv(root_dir+"/Final Model/"+ExampleName)

    dataTimeStamps = []
    dataValues = []

    for column in data:
        if "time" in column:
            dataTimeStamps = data[column].to_numpy()
            data = data.drop(columns=column)
            continue
        if not "input" in column:
            data = data.drop(columns=column)

    dataValues = data.to_numpy()

    if len(timeStamps) == 0:
        #no time stamps
        for i in range(0,dataValues.shape[-1]):
            ax.plot(data[...,i],label="Dim. "+str(i))
    else:
        #Data has Timestamps
        for i in range(0,dataValues.shape[-1]):
            ax.plot(x=dataTimeStamps,y=data[...,i],label="Dim. "+str(i))
    
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Values")
    ax.set_title("Data of "+ExampleName)

if __name__ == "__main__":

    if len(sys.argv)< 3:
        raise ValueError("Pleas Specify the root dir and the name of the example you want to plot")

    rootDir = sys.argv[1]
    ExampleName = sys.argv[2]

    img,ax = plt.subplots()

    plotExample(rootDir,ax,ExampleName)

    plt.show()
    plt.close()
