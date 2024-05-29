# Deep Learning for Detecting and Early Predicting Chronic Obstructive Pulmonary Disease from Spirogram Time Series: A UK Biobank Study
## Overview
This repository contains code developed as part of the following paper: "Deep Learning for Detecting and Early Predicting Chronic Obstructive Pulmonary Disease from Spirogram Time Series: A UK Biobank Studys"

## Installation
Installation supports Python 3.8 and Pytorch 2.2.2. You can use the following commands to conveniently set up the environment after pulling the repository:
```bash
conda create -n DeepSpiro python=3.8
conda activate DeepSpiro
pip install -r requirements.txt
```

## Before Running
Before you run the code, please run the `generate_example_data.py` file. This file will download the Spirometry exhalation volume curve example from the UK Biobank website and generate the data needed to run the test program. If you obtained approval for the UK Biobank dataset, you can access Data-Field 3062 (FVC), Data-Field 3063 (FEV1), Data-Field 3064 (PEF), and Data-Field 3066 (flow). And populate the sample.xlsx file in the data directory with the data you obtained. Note that our `generate_example_data.py` file just generates FVC, FEV1, and PEF by code, and may not be as accurate as the UK Biobank data.


## Run Model Predict
```bash
conda activate DeepSpiro
python run_predict.py
```

## Data Sources
Data from the UK Biobank, which is available after the approval of an application at https://www.ukbiobank.ac.uk.