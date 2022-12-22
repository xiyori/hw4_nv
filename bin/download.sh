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
gdown https://drive.google.com/u/0/uc?id=1tr9pKTnF1L5P1XGEQpmv9DnOLczpo7L5
mv v1_l1mel45_genlr0.0002_dislr0.0002_batch16.chk "$2"

#download results
gdown https://drive.google.com/u/0/uc?id=1O_2Pt0kk5qieVlXBP1YCDDK9QqDhcqEA
unzip inference_audio.zip >> /dev/null
mv inference_audio "$3"
