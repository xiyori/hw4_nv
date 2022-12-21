#!/bin/bash

#download LjSpeech
echo "download LjSpeech"
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
mv LJSpeech-1.1/* "$1"

#download labels
echo "download labels"
wget https://github.com/jik876/hifi-gan/raw/master/LJSpeech-1.1/training.txt -o /dev/null
wget https://github.com/jik876/hifi-gan/raw/master/LJSpeech-1.1/validation.txt -o /dev/null
mv training.txt "$1"
mv validation.txt "$1"

#download checkpoint
#gdown https://drive.google.com/u/0/uc?id=1VVyVwQxDtKyWCvpMgI069V5tZJRIuKAO
#mv final_model.pth.tar "$2"
#
##download results
#gdown https://drive.google.com/u/0/uc?id=1YusPal6ao80Ou8sRSNZSEAljOA7gmMSc
#unzip final_model.zip >> /dev/null
#mv final_model "$3/final_model"
