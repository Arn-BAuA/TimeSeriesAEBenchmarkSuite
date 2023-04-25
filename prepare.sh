#!/bin/bash

#Shellscript to prepare the system for the framework (create virtual environment, download datasets, ect.)

#packages needet: python...
#python is python3

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

kaggle datasets download -d 
