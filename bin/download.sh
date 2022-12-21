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
gdown https://drive.google.com/u/0/uc?id=12WleeyYBxTWAJHJtlb5fC-XvRQmid57B
mv v1_l1mel45_genlr0.0002_dislr0.0002_batch16_epoch8.pth "$2/../models"

#download results
gdown https://drive.google.com/u/0/uc?id=1Ej-N42vodpVAYFXQ8RX5yDSVF1D8qsVg
unzip v1_l1mel45_genlr0.0002_dislr0.0002_batch16_epoch8.zip >> /dev/null
mv v1_l1mel45_genlr0.0002_dislr0.0002_batch16_epoch8 "$3"
