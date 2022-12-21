# Homework 4 (NV)

## Task
This homework is aimed at implementing [HiFiGAN](https://arxiv.org/pdf/2010.05646.pdf) vocoder.
We understand that anyone can find the authors' source code, but please don't write it off mindlessly. (At least the reviewer will see it and penalize you)

Use dataset [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) you already know.

NV homework repository of HSE DLA course. The goal of the project is to implement HiFi-GAN model and train it on LJ-Speech dataset.

### Installation

To install necessary python packages run the command:

`pip install -r requirements.txt`

### Resources

Download all needed resources (data, checkpoints & inference examples) with

`python3 bin/download.py`

If you use Yandex DataSphere, specify the option

`python3 bin/download.py -d`

**One may use** `datasphere.ipynb` **notebook that contains all necessary commands to reproduce the results of the project.**

### Training

Once the resources are ready, start the training with

`python3 train.py`

for DataSphere `g1.1` configuration.

### Generating test audio

Use

`python3 inference.py`

to generate test audio files.

The audio is stored at `resources/predicted`.

### General usage

Run `python3 some_script.py -h` for help.

