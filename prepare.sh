#!/bin/bash

#Shellscript to prepare the system for the framework (create virtual environment, download datasets, ect.)

#packages needet: python...
#python is python3
#svn
sudo apt-get install python python3 subversion

#create venv:
VenvFolder="fwVenv"

if ! [-d "$VenvFolder"]; then
	python -m venv "$VenvFolder"
fi

source "$fwVenv/bin/activate"

pip install numpy
pip install pandas
pip install matplotlib
pip install pytorch
#for downloading some datasets
pip install kaggle

#make necessairy dirs...

#download Datasets...
dataPath="data/"
currentPath=$(pwd)

cd dataPath

#ECG Dataset
kaggle datasets download -d shayanfazeli/heartbeat
mkdir ECG
unzip heartbeat.zip -d ECG
rm heartbeat.zip

#SMD
svn export https://github.com/NetManAIOps/OmniAnomaly/trunk/ServerMachineDataset
mv ServerMachineDataset SMD

#Smap And MSL (Code "inspired" by OmniAnomaly Repo: https://github.com/NetManAIOps/OmniAnomaly)
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip
mkdir SMAP_and_MSL
unzip data.zip -d SMAP_and_MSL
rm data.zip
wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
mv labeled_anomalies.csv SMAP_and_MSL/

#SWaT
#you have to add by your self
mkdir SWaT

#UCR
read -sp "Pleas enter the UCR Zip-Password (it can be found on : https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)" UCRPassWD
wget https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip
mkdir UCR
unzip -P $UCRPassWD UCRArchive_2018.zip -d UCR
rm UCRArchive_2018.zip

#Engine
kaggle datasets download behrad3d/nasa-cmaps
mkdir Engine
unzip nasa-cmaps.zip -d Engine
rm nasa-cmaps.zip

cd currentPath
