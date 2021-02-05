# TLS Mal Detect
Using machine learning to detect malware in encrypted TLS traffic metadata

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Linux](https://img.shields.io/badge/Supports-Linux-green.svg)
![Windows](https://img.shields.io/badge/Supports-Windows-green.svg)
![Docker](https://img.shields.io/badge/Supports-Docker-red.svg)
![Python: v3.6](https://img.shields.io/badge/Python-3.6-blue.svg)
[![Python: Reference](https://img.shields.io/badge/Python-Reference-blue.svg)](https://docs.python.org/3.9/)
---

The purpose of this repository is to evaluate multiple machine learning algorithms and demonstrate their ability to accurately classify malicious traffic. Three models were used in this research, a One-Class Support Vector Machine, a Support Vector Machine, and an Autoencoder Neural Network. This repository is supplied as a part of a research assignment in support of my Master's of Science in Information Security Engineering from SANS Technology Institute entitled *Malware Detection in Encrypted TLS Traffic Through Machine Learning*.

---

> NOTE: If you decide to use this program with Windows, there are issues with pathlib. It works best if you convert the pathlib paths to raw strings.

## Run from Docker

The easiest way to use this is to use Docker.
1. Download repository (git clone https://github.com/1computerguy/tls-mal-detect)
2. cd to tls-mal-detect directory
3. Unzip the file `test-train-data/test_train_data.7z`
4. Build the container with Docker:
    - Change directory to the tls-mal-detect repo directory
    - Run the docker build command
    - Run the docker container

```
docker build . --tag tls-mal-detect:latest
docker run --rm -it tls-mal-detect
```

The `docker run` command above will provide the script help documentation below:
```
usage: anomaly-detect.py [-h] -d DATA_SIZE [-m MALWARE_SIZE] [-t TEST_SIZE] [-o ML_MODEL] [-s] [-l] [-f FILE] [-r] [-g GRAPH] [-p] [-e]

Run an ML model to analyse TLS data for malicious activity.

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_SIZE, --data DATA_SIZE
                        Data sample size to analyze
  -m MALWARE_SIZE, --malware MALWARE_SIZE
                        Percentage of dataset that is Malware
  -t TEST_SIZE, --test TEST_SIZE
                        Percentage of dataset to use for validation
  -o ML_MODEL, --model ML_MODEL
                        Machine Learning model to use. Acceptable values are:
                        - ae = Autoencoder
                        - svm = Support Vector Machine
                        - oc-svm = One-Class SVM
  -s, --save            Save the trained model - REQUIRES the -f/--file option
  -l, --load            Evaluate data against a trained model - REQUIRES the -f/--file option
  -f FILE, --file FILE  Save/Load file path
  -r, --scores          Print 10-fold cross-validated Accuracy, Recall, Precision, and F2 scores
  -g GRAPH, --graph GRAPH
                        Visualize the modeled dataset. Acceptable values are:
                        SVM and OC-SVM graphs:
                            - confusion
                            - margin (SVM only)
                            - boundary (SVM only)
                            - auc
                        Autoencoder graphs:
                            - confusion
                            - loss
                            - mae
                            - thresh
                            - scatter
  -p, --print           Print dataset
  -e, --export          This will save the graph to a file - REQUIRED if running in a container
```

---

#### Examples running the script from Docker


Train the One-Class SVM with a 25,000 sample dataset, a 1% malware distribution, a 20% Validation dataset, and print the cross-validated accuracy, precision, recall, and F2-scores

```
docker run --rm -it tls-mal-detect -d 25000 -m 1 --model oc-svm --scores
```

Run the SVM using the pre-saved model

```
docker run --rm -it tls-mal-detect -d 50000 -m 20 --model svm --scores --load --file /detect/models/svm.pkl
```

Run the Autoencoder and export the Confusion Matrix graph as a .png file to the current working directory of the host or VM

```
docker run --rm -it -v $(pwd):/detect/graph dtls-mal-detect -d 25000 -m 5 --model ae --graph confusion
```

---

## Run from Host (or VM) - *Linux recommended*
1. Download repository (git clone https://github.com/1computerguy/tls-mal-detect)
2. cd to tls-mal-detect directory
3. Unzip the file `test-train-data/test_train_data.7z`
4. Install Python3 and requirements
    - Install python3 according to your Operating System's requirements
    - Install pip
    - Use pip to install additional requirements `pip install -r requirements.txt`
    - Run the program

Print script help:

```
python3 anomaly-detect.py -h
```

Train the One-Class SVM with a 25,000 sample dataset, a 1% malware distribution, a 20% Validation dataset, and print the cross-validated accuracy, precision, recall, and F2-scores

```
python3 anomaly-detect.py -d 25000 -m 1 --model oc-svm --scores
```

Run the SVM using the pre-saved model

```
python3 anomaly-detect.py -d 50000 -m 20 --model svm --scores --load --file ./models/svm.pkl
```

Run the Autoencoder and open the Confusion Matrix graph

```
python3 anomaly-detect.py -d 25000 -m 5 --model ae --graph confusion
```

Run the SVM and save the margin graph to disk

```
python3 anomaly-detect.py -d 25000 -m 20 --model svm --graph margin --export
```
