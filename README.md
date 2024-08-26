# MF-Baynet
This project involves training Bayesian Neural Networks (BNNs) using a dataset related to aerodynamic coefficients (e.g., cl, cm). The project provides scripts and utilities to train, fine-tune, and evaluate BNN models using data of varying fidelities.

## Introduction
This repository contains code to train and evaluate Bayesian Neural Networks (BNNs) on a dataset related to aerodynamic studies. The models are trained on data with different fidelities (e.g., low, medium, high), and the script supports fine-tuning and transfer learning.

## Features
Multi-Fidelity Training: Train BNNs using data of varying fidelities.
Transfer Learning: Fine-tune models on different fidelity levels.
Model Saving/Loading: Save and load trained models for later use.
Visualization: Generate plots to visualize training results.

## Installation
To use this project, you'll need to have Python installed along with the necessary dependencies. The main dependencies include:

torch
numpy
pandas
matplotlib
You can install these dependencies using pip:
pip install torch numpy pandas matplotlib

## Usage
### Preparing the Dataset
Ensure that your dataset is located in the Dataset/ directory with the correct format. The script expects a CSV file with specific columns:

mach
aoa
cl
cm
fidelity (e.g., "low", "mid", "cfd")

### Flags in the Script
TRAIN_BNN_LF_FLAG: Train BNN on low-fidelity data.
TRAIN_BNN_MF_FLAG: Train BNN on medium-fidelity data.
TRAIN_BNN_HF_FLAG: Train BNN on high-fidelity data.
TRAIN_TL_FLAG: Perform transfer learning.
TRAIN_TL_FT_FLAG: Fine-tune the transfer learning model.
SAVE_MODEL_FLAG: Save the trained model to a specified directory.

### Saving and Loading Models
To save the model after training, set the SAVE_MODEL_FLAG to True. The model will be saved in the directory specified by the study_path variable.

### Results
After training, results such as plots and model evaluations will be saved in the AIModels/ directory. You can visualize these results using the provided scripts or your preferred tools.
