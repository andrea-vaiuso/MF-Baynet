import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import torchbnn as bnn
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
import re
import scipy
import random
from utils import PrCol
from bnn import BNN, BNNDataset
import copy
import os

Color = PrCol()

TRAIN_BNN_LF_FLAG = False
TRAIN_BNN_MF_FLAG = False
TRAIN_BNN_HF_FLAG = False
TRAIN_TL_FLAG = False
TRAIN_TL_FT_FLAG = False

SAVE_MODEL_FLAG = True

dataset_location = "Dataset/bscw_dataset.csv"
SEP = ";"
input_labels = ["mach","aoa"]
output_labels = ["cl","cm"]
fidelity_column_name = "fidelity"
fidelities = ["low","mid","cfd"]

model_name_prefix = "BSCW"
dataset_all = pd.read_csv(dataset_location,sep=SEP)
dataset_lf_denorm = dataset_all[dataset_all[fidelity_column_name] == "low"]
dataset_mf_denorm = dataset_all[dataset_all[fidelity_column_name] == "mid"]
dataset_hf_denorm = dataset_all[dataset_all[fidelity_column_name] == "cfd"]
dataset_all
from scalers import MinMaxScaler as MMS

scaler = MMS(dataset_all.drop(columns=[fidelity_column_name], inplace=False),interval=(1,2))
normalized_datasets = []
for fid in fidelities:
    dataset_fidelity = dataset_all[dataset_all[fidelity_column_name] == fid]
    dataset_fidelity_norm = scaler.scaleDataframe(dataset_fidelity.drop(columns=[fidelity_column_name], inplace=False))
    normalized_datasets.append(dataset_fidelity_norm)
scaler.save(f"NormalizationData/{model_name_prefix}norm.pkl")

dataset_lf = BNNDataset(normalized_datasets[0],input_labels,output_labels)
dataset_mf = BNNDataset(normalized_datasets[1],input_labels,output_labels)
dataset_hf = BNNDataset(normalized_datasets[2],input_labels,output_labels)
dataset_all = BNNDataset(pd.concat([normalized_datasets[0], normalized_datasets[1], normalized_datasets[2]], axis=0),input_labels,output_labels)

train_lf, valid_lf = dataset_lf.train_val_split()
train_mf, valid_mf = dataset_mf.train_val_split()
train_hf, valid_hf, test_hf = dataset_hf.train_val_test_split(train_size=0.11,val_size=0.2,seed=42)
model_path = f"AIModels/{model_name_prefix}_bnn_lf.pt"

bnn_lf_model = BNN(
    in_dim=len(input_labels),
    out_dim=len(output_labels),
    mu = 0,
    std = 0.1176,
    units = [128,128],
    denseOut = False,
    dropout = False,
    device=torch.device("cpu"),
    activation=nn.LeakyReLU(),
    model_name="BNN LF"
)
if TRAIN_BNN_LF_FLAG:
    bnn_lf_model.train(
        train_lf,
        valid_lf,
        patience=150,
        n_epochs=5000,
        batch_size = 32
    )
    if SAVE_MODEL_FLAG: bnn_lf_model.save(model_path)
else: bnn_lf_model.load(model_path)
    

model_path = f"AIModels/{model_name_prefix}_bnn_mf.pt"

bnn_mf_model = BNN(
    in_dim=len(input_labels),
    out_dim=len(output_labels),
    mu = 0,
    std = 0.1176,
    units = [128,128],
    denseOut = False,
    dropout = False,
    device = torch.device("cpu"),
    activation = nn.LeakyReLU(),
    model_name = "BNN MF"
)
if TRAIN_BNN_MF_FLAG:
    bnn_mf_model.train(
        train_mf,
        valid_mf,
        patience=500,
        n_epochs=10000,
        batch_size = 8
    )
    if SAVE_MODEL_FLAG: bnn_mf_model.save(model_path)
else: bnn_mf_model.load(model_path)

model_path = f"AIModels/{model_name_prefix}_bnn_hf.pt"

bnn_hf_model = BNN(
    in_dim=len(input_labels),
    out_dim=len(output_labels),
    mu = 0,
    std = 0.1176,
    units = [128,128],
    denseOut = False,
    dropout = False,
    device = torch.device("cpu"),
    activation = nn.LeakyReLU(),
    model_name = "BNN HF"
)
if TRAIN_BNN_HF_FLAG:
    bnn_hf_model.train(
        train_hf,
        valid_hf,
        patience=1000,
        n_epochs=100000,
        batch_size = 8
    )
    if SAVE_MODEL_FLAG: bnn_hf_model.save(model_path)
else: bnn_hf_model.load(model_path)
bnn_df_model = copy.deepcopy(bnn_lf_model)
bnn_df_model.model_name = "BNN TL"

model_path_1 = f"AIModels/{model_name_prefix}_bnn_tl_2.torch"
model_path_2 = f"AIModels/{model_name_prefix}_bnn_tl_finetune_2.torch"

if TRAIN_TL_FLAG:
    params_to_freeze = ["weight_mu", "weight_log_sigma", "bias_mu", "bias_log_sigma"]
    layers_to_freeze = ["in_layer", "hidden_layers.0", "hidden_layers.1", "out_layer"]
    #FREEZE ALL THE PARAMS IN THE MODEL
    bnn_df_model.setModelGradients(False)
    bnn_df_model.setModelGradients(True,layers=["hidden_layers.0","out_layer"])
    #Train with MF
    bnn_df_model.train(
        train_mf,
        valid_mf,
        n_epochs=8000,
        lr=0.001,
        restoreBestModel=True,
        patience=500,
        batch_size = 32,
        earlyStopping=True
    )
    if SAVE_MODEL_FLAG: bnn_df_model.save(model_path_1)
else: 
    bnn_df_model.load(model_path_1)
    bnn_df_model_fine_tuned = copy.deepcopy(bnn_df_model) 
    bnn_df_model_fine_tuned.model_name = "BNN TL FT"

if TRAIN_TL_FT_FLAG:
    bnn_df_model_fine_tuned = copy.deepcopy(bnn_df_model) 
    bnn_df_model_fine_tuned.model_name = "BNN TL FT"
    bnn_df_model.setModelGradients(False)
    bnn_df_model.setModelGradients(True,layers=["out_layer"])
    bnn_df_model_fine_tuned.train(
        train_hf,
        valid_hf,
        n_epochs=8000,
        lr=0.001,
        restoreBestModel=True,
        patience=500,
        batch_size = 32,
        earlyStopping=True
    )
    if SAVE_MODEL_FLAG: bnn_df_model_fine_tuned.save(model_path_2)
else: 
    bnn_df_model_fine_tuned.load(model_path_2)

all_dataset_lf = normalized_datasets[0].assign(fidelity='lf')
all_dataset_mf = normalized_datasets[1].assign(fidelity='mf')
all_dataset_hf = normalized_datasets[2].assign(fidelity='hf')

all_fid_df_norm = pd.concat([all_dataset_lf,all_dataset_mf,all_dataset_hf],axis=0)

asd = all_fid_df_norm[all_fid_df_norm["fidelity"] == "hf"].groupby(['mach'])['aoa'].nunique().reset_index(name='unique_aoa_count')
asd.iloc[-1].values[0]

index_value = 0
value_to_show = output_labels[index_value]

n_attempt = 100

from matplotlib import colors

num = 5
mach_values = np.linspace(0.8, 2.2, num=num)
aoa_values = np.linspace(1, 2.2, num=num)
mach_values_denorm = scaler.reverseArray([mach_values], columns=["mach"])[0]
aoa_values_denorm = scaler.reverseArray([aoa_values], columns=["aoa"])[0]
input_space = np.array([(mach, aoa) for mach in mach_values for aoa in aoa_values])
input_space_denorm = np.array([(mach, aoa) for mach in mach_values_denorm for aoa in aoa_values_denorm])
pred_mean, pred_std = bnn_df_model_fine_tuned.predict(input_space,scaler=scaler,output_labels=output_labels,attempt=n_attempt)
pred_mean = np.array(pred_mean)[:,0]
pred_std = np.array(pred_std)[:,0]
pred_upper = pred_mean + pred_std
pred_lower = pred_mean - pred_std

train_hf_denorm = scaler.reverseDataframe(train_hf.data)

# Creare il grafico 3D
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Creare la superficie 3D
mach_grid, aoa_grid = np.meshgrid(mach_values_denorm, aoa_values_denorm)
pred_mean_grid = pred_mean.reshape((num, num))
pred_upper_grid = pred_upper.reshape((num, num))
pred_lower_grid = pred_lower.reshape((num, num))

ax.plot_surface(mach_grid.T, aoa_grid.T, pred_mean_grid, color="blue", alpha=0.5)
ax.plot_surface(mach_grid.T, aoa_grid.T, pred_upper_grid, color="red", alpha=0.2)
ax.plot_surface(mach_grid.T, aoa_grid.T, pred_lower_grid, color="red", alpha=0.2)
ax.scatter(dataset_hf_denorm["mach"],dataset_hf_denorm["aoa"], dataset_hf_denorm["cl"], c='r', marker='o', label="High Fidelity")
ax.scatter(train_hf_denorm["mach"],train_hf_denorm["aoa"], train_hf_denorm["cl"], c='r', marker='o', s=100, label="High Fidelity_train")
ax.scatter(dataset_mf_denorm["mach"],dataset_mf_denorm["aoa"], dataset_mf_denorm["cl"], c='g', marker='o', label="Mid Fidelity")
ax.scatter(dataset_lf_denorm["mach"],dataset_lf_denorm["aoa"], dataset_lf_denorm["cl"], c='y', marker='o', label="Low Fidelity", alpha=0.4)


# Etichettare gli assi
ax.set_xlabel('Mach')
ax.set_ylabel('AOA')
ax.set_zlabel('Pred Mean')


# Aggiungere la legenda
# fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Mostrare il grafico
plt.legend()
plt.show()