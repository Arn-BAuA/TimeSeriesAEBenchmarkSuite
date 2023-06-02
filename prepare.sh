#!/bin/bash

#Shellscript to prepare the system for the framework (create virtual environment, download datasets, ect.)

#packages needet: python...
#python is python3
#svn
sudo apt-get install -y subversion

#create venv:
VenvFolder="fwVenv"

if ! [ -d "$VenvFolder" ]; then
	python3 -m venv "$VenvFolder"
fi

source "$VenvFolder/bin/activate"

pip install numpy
pip install pandas
pip install matplotlib
pip install torch
#for downloading some datasets
pip install kaggle

#make necessairy dirs...

#download Datasets...
dataPath="data/"
mkdir -p $dataPath
currentPath=$(pwd)

cd $dataPath

#ECG Dataset
ECGDir="ECG"
if ! [ -d "$ECGDir" ]; then
	kaggle datasets download -d shayanfazeli/heartbeat
	mkdir $ECGDir
	unzip heartbeat.zip -d ECG
	rm heartbeat.zip
fi

#SMD
SMDDir="SMD"

if ! [ -d "$SMDDir" ]; then
	svn export https://github.com/NetManAIOps/OmniAnomaly/trunk/ServerMachineDataset
	mv ServerMachineDataset $SMDDir
fi

#Smap And MSL (Code "inspired" by OmniAnomaly Repo: https://github.com/NetManAIOps/OmniAnomaly)
SMAPDir="SMAP_and_MSL"
if ! [ -d "$SMAPDir" ]; then
	wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip
	mkdir $SMAPDir
	unzip data.zip -d $SMAPDir
	rm data.zip
	wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
	mv labeled_anomalies.csv $SMAPDir/
fi

#SWaT
#you have to add by your self
mkdir -p SWaT
echo ""
echo ""
echo ""
echo ""
echo ""
echo ""
echo ""
read -sp "We created the SWaT folder for you. You have to request them by filling in the following form: https://docs.google.com/forms/d/e/1FAIpQLSdwOIR-LuFnSu5cIAzun5OQtWXcsOhmC7NtTbb-LBI1MyOcug/viewform. Press Enter to Continue." SomeResponse

#UCR
UCRDir="UCR"
if ! [ -d "$UCRDir" ]; then
	read -sp "Pleas enter the UCR Zip-Password (it can be found on : https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)" UCRPassWD
	wget https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip
	mkdir $UCRDir
	unzip -P $UCRPassWD UCRArchive_2018.zip -d $UCRDir
	rm UCRArchive_2018.zip
fi

#Engine
EngineDir="Engine"
if ! [ -d "$EngineDir" ]; then
	kaggle datasets download behrad3d/nasa-cmaps
	mkdir $EngineDir
	unzip nasa-cmaps.zip -d $EngineDir
	rm nasa-cmaps.zip
fi

cd $currentPath
