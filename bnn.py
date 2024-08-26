# AUTHOR: Andrea Vaiuso
MODEL_VERSION = "4.14"

import torch
import torch.nn as nn
import torchbnn as bnn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from utils import PrCol, CircularBuffer, LoadBar, seconds_to_hhmmss
from time import time
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from scalers import MinMaxScaler as MMS
from torch.utils.data import Dataset
import pandas as pd
import os
from typing import List, Tuple

Color = PrCol()

class BNNDataset(Dataset):
    
    def __init__(self, data: pd.DataFrame, input_labels: list, output_labels: list, device = "cpu", dtype = torch.float32):
        self.x = data[input_labels]
        self.y = data[output_labels]
        self.data = data
        self.input_labels = input_labels
        self.output_labels = output_labels
        self.device = device
        self.dtype = dtype

    def __add__(self, other):
        if not isinstance(other, BNNDataset):
            raise TypeError("Both operands must be of type BNNDataset")
        combined_data = pd.concat([self.data, other.data], ignore_index=True)
        return BNNDataset(combined_data, 
                          self.input_labels, 
                          self.output_labels, 
                          self.device, 
                          self.dtype)

    def __str__(self):
        return self.data.__str__()
    
    def __sizeof__(self) -> int:
        return self.data.__sizeof__()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x.iloc[idx].values, device=self.device, dtype=self.dtype), torch.tensor(self.y.iloc[idx].values, device=self.device, dtype=self.dtype)
    
    def train_val_test_split(self, train_size: float = 0.7, val_size: float = 0.15, seed: int = 42):
        # Calcola le lunghezze per train, validation e test
        total_size = len(self.data)
        train_len = int(train_size * total_size)
        val_len = int(val_size * total_size)

        # Shuffle del dataset
        data = self.data.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Dividi il dataset in train, validation e test
        train_data, val_data, test_data = data[:train_len], data[train_len:train_len + val_len], data[train_len + val_len:]
        print(train_data)
        # Crea gli oggetti BNNDataset per train, validation e test
        train_dataset = BNNDataset(train_data, self.input_labels, self.output_labels)
        val_dataset = BNNDataset(val_data, self.input_labels, self.output_labels)
        test_dataset = BNNDataset(test_data, self.input_labels, self.output_labels)

        return train_dataset, val_dataset, test_dataset
    
    def train_val_split(self, train_size: float = 0.7, seed: int = 42):
        # Calcola le lunghezze per train, validation e test
        total_size = len(self.data)
        train_len = int(train_size * total_size)

        # Shuffle del dataset
        data = self.data.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Dividi il dataset in train, validation e test
        train_data, val_data = data[:train_len], data[train_len:]

        # Crea gli oggetti BNNDataset per train, validation e test
        train_dataset = BNNDataset(train_data, self.input_labels, self.output_labels)
        val_dataset = BNNDataset(val_data, self.input_labels, self.output_labels)

        return train_dataset, val_dataset

class BNN(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 mu=0, 
                 std=0.5, 
                 units=[100], 
                 denseOut = False,
                 dropout = False,
                 device = "cpu",
                 activation = nn.LeakyReLU(),
                 model_name = "BNN DEFAULT"
                 ):
        super().__init__()
        self.version = MODEL_VERSION
        self.model_name = model_name
        self.device = device
        self.to(self.device)
        self.dropout_flag = dropout
        self.activation = activation.to(self.device)
        
        if self.dropout_flag: self.dropout_norm = nn.Dropout(0.2)
        self.in_layer = bnn.BayesLinear(prior_mu=mu, prior_sigma=std, in_features=in_dim, out_features=units[0]).to(self.device)
        self.hidden_layers = nn.ModuleList([
            bnn.BayesLinear(prior_mu=mu, prior_sigma=std, in_features=units[i-1], out_features=units[i]).to(self.device)
            for i in range(1,len(units))
        ])
        if self.dropout_flag: self.hidden_dropout = nn.ModuleList([
            nn.Dropout(0.2)
            for _ in range(1,len(units))
        ])
        if denseOut:
            self.out_layer = nn.Linear(units[-1],out_dim).to(self.device)
        else:
            self.out_layer = bnn.BayesLinear(prior_mu=mu, prior_sigma=std, in_features=units[-1], out_features=out_dim).to(self.device)

    def forward(self, x):
        x = self.activation(self.in_layer(x))
        if self.dropout_flag: x = self.dropout_norm(x)
        for i in range(len(self.hidden_layers)):
            layer = self.hidden_layers[i]
            if self.dropout_flag: dropout = self.hidden_dropout[i]
            x = self.activation(layer(x))
            if self.dropout_flag: x = dropout(x)
        return self.out_layer(x)
    
    def train(self, 
            train_data: BNNDataset,
            valid_data: BNNDataset,
            n_epochs: int = 1000, 
            patience: int = 20, 
            lr: float = 0.001, 
            batch_size:int = 1, 
            earlyStopping:bool = True, 
            shuffle:bool = True, 
            restoreBestModel:bool = True, 
            verbose:bool = True, 
            show_loadbar:bool = True,
            plotHistory:bool = True,
            lrdecay:tuple = None, #(step_size, gamma)
            lrdecay_limit:float = 0.00005,
            history_plot_saving_path: str = None,
            batchnorm:int = 0
            ):
        self.earlyStopping = earlyStopping
        train_data.device = self.device
        valid_data.device = self.device
        if verbose: print(f"{Color.yellow}Creating dataloader (train size: {len(train_data)}, valid size: {len(valid_data)})...{Color.end}")
        train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        valid_data_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=shuffle)
        if verbose: print(f"{Color.yellow}Initializing model architecture...{Color.end}")
        if verbose: print(self)
        if not self.earlyStopping:
            patience = np.inf
        
        self.kl_weight = 1 / len(train_data)
        self.mse_loss = nn.MSELoss().to(self.device)
        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False).to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        if lrdecay:
            if isinstance(lrdecay, tuple): 
                if len(lrdecay) == 2: 
                    self.scheduler = StepLR(self.optimizer, step_size=lrdecay[0], gamma=lrdecay[1]) #30, 0.9
                else: raise TypeError("lrdecay parameter must be a bidimensional tuple: (step_size, gamma)")
            else: raise TypeError("lrdecay parameter must be a bidimensional tuple: (step_size, gamma)")
        else: self.scheduler = None

        self.best_val_loss = np.inf
        patience_count = 0
        best_model = self
        self.best_epoch = np.nan
        self.train_loss = np.inf
        self.valid_loss = np.inf
        self.train_loss_history = []
        self.valid_loss_history = []
        self.timeBuffer = CircularBuffer(50)
        loadBar = LoadBar()
        n_grd_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_tot_params = sum(p.numel() for p in self.parameters())
        if verbose: print(f"{Color.green}[{Color.end}{self.model_name}{Color.green}] >> Model initialized ({n_grd_params} learnable / {n_tot_params} total parameters){Color.end}\n")
        if verbose: print(f"{Color.green}Start Training{Color.end}\n")
        try:
            tot_t1 = time()
            for epoch in range(n_epochs):
                self.actual_epoch = epoch
                loadBar.tick()
                t1 = time()
                current_train_loss = 0
                current_valid_loss = 0

                with torch.no_grad():
                    if batchnorm==1 or batchnorm==2:
                        for param in self.parameters():
                            param.data = F.normalize(param.data, p=batchnorm, dim=0)

                if show_loadbar: print(f'{loadBar.loadBar(epoch + 1, n_epochs)} ' +
                        f'MSE (TRAIN) : {Color.yellow}{self.train_loss:.6f}{Color.end}, ' +
                        f'MSE (VAL) : {Color.cyan}{self.valid_loss:.6f}{Color.end} -- ' +
                        f'BEST Val Loss : {Color.green}{self.best_val_loss:.6f}{Color.end} ' +
                        f'(at epoch {self.best_epoch}) ' +
                        (f'Eary Stopping in: {str(patience - patience_count).zfill(4)}   ' if self.earlyStopping else '    '), end="\r")

                for i, data in enumerate(train_data_loader):
                    x,y = data
                    pre = self(x)
                    mse = self.mse_loss(pre, y)
                    kl = self.kl_loss(self)
                    cost = mse + self.kl_weight * kl

                    self.optimizer.zero_grad()
                    cost.backward()
                    self.optimizer.step()
                    current_train_loss += cost.item()

                self.train_loss = current_train_loss

                # Validation loop with batches
                for i, data in enumerate(valid_data_loader):
                    x,y = data
                    pre = self(x)
                    mse = self.mse_loss(pre, y)
                    kl = self.kl_loss(self)
                    cost = mse + self.kl_weight * kl
                    current_valid_loss += cost.item()

                self.valid_loss = current_valid_loss

                if self.optimizer.param_groups[0]['lr'] > lrdecay_limit and lrdecay:
                    self.scheduler.step()

                if self.valid_loss <= self.best_val_loss:
                    self.best_val_loss = self.valid_loss
                    patience_count = 0
                    best_model = self
                    self.best_epoch = epoch
                else:
                    patience_count += 1

                self.train_loss_history.append(self.train_loss)
                self.valid_loss_history.append(self.valid_loss)
                t2 = time() - t1
                self.timeBuffer.add_element(t2)
                loadBar.tock()
                if patience_count >= patience:
                    if verbose: print(f"\n{Color.yellow}Early stopping at epoch {self.best_epoch}{Color.end}")
                    if verbose: print(f"Total enlapsed time: {(time() - tot_t1):.2f} sec")
                    break

            if plotHistory: self._plotHistory(save_path=history_plot_saving_path)
            if restoreBestModel:
                if verbose: print(f"{Color.green}Saving best model at epoch {self.best_epoch}{Color.end}")
                self = best_model
            if verbose: print(f"Total enlapsed time: {(time() - tot_t1):.2f} sec")
            return 

        except KeyboardInterrupt:
            if verbose: print(f"\n{Color.magenta}Interrupting training at epoch {self.best_epoch}...{Color.end}")
            if plotHistory: self._plotHistory(save_path=history_plot_saving_path)
            if restoreBestModel:
                if verbose: print(f"{Color.green}Saving best model at epoch {self.best_epoch}{Color.end}")
                self = best_model
            if verbose: print(f"Total enlapsed time: {(time() - tot_t1):.2f} sec")
            return
        
    def k_fold_cross_validation(self, train_data, k=5, **train_args):
        kf = KFold(n_splits=k)
        fold_train_losses = []
        fold_valid_losses = []

        for train_idx, valid_idx in kf.split(train_data):
            train_fold = torch.utils.data.Subset(train_data, train_idx)
            valid_fold = torch.utils.data.Subset(train_data, valid_idx)

            self.reset()  # Reset model parameters for each fold

            # Train the model on the current fold
            self.train(train_fold, valid_fold, **train_args)

            # Save training and validation losses for this fold
            fold_train_losses.append(self.train_loss_history)
            fold_valid_losses.append(self.valid_loss_history)

        # Aggregate losses across folds
        avg_train_losses = np.mean(np.array(fold_train_losses), axis=0)
        avg_valid_losses = np.mean(np.array(fold_valid_losses), axis=0)

        return avg_train_losses, avg_valid_losses
    
    def getAllParametersName(self):
        par_name = []
        for name, _ in self.named_parameters():
            name_splitted = name.split(".")
            if len(name_splitted) > 2:
                name_splitted = [name_splitted[0] + "." + name_splitted[1], name_splitted[2]]
            par_name.append(name_splitted[-1])
        return list(set(par_name))
    
    def getAllLayersName(self):
        par_name = []
        for name, _ in self.named_parameters():
            name_splitted = name.split(".")
            if len(name_splitted) > 2:
                name_splitted = [name_splitted[0] + "." + name_splitted[1], name_splitted[2]]
            par_name.append(name_splitted[0])
        return list(dict.fromkeys(par_name))

    def setModelGradients(self, 
                          requires_grad, 
                          params = None,
                          layers = None):
        if params is None: params = self.getAllParametersName()
        if layers is None: layers = self.getAllLayersName()
        for name, param in self.named_parameters():
            name_splitted = name.split(".")
            if len(name_splitted) > 2:
                name_splitted = [name_splitted[0] + "." + name_splitted[1], name_splitted[2]]
            if name_splitted[-1] in params and name_splitted[0] in layers:
                param.requires_grad = requires_grad
    
    def _plotHistory(self,save_path=None):
        lim_max = max([max(self.train_loss_history), max(self.valid_loss_history)])
        lim_min = min([min(self.train_loss_history), min(self.valid_loss_history)])
        half = (lim_max - lim_min) / 2

        x = range(len(self.train_loss_history))
        min_y_val = min(self.valid_loss_history)

        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        axs[0].plot(x, self.train_loss_history, color="orange", label="train loss")
        axs[0].plot(x, self.valid_loss_history, color="blue", label="valid loss")
        
        if self.earlyStopping:
            axs[0].axvline(self.best_epoch, linestyle="--", color="red")
        
        axs[1].plot(x, self.train_loss_history, color="orange",  label="train loss")
        axs[1].plot(x, self.valid_loss_history, color="blue", label="valid loss")

        if self.earlyStopping:
            axs[1].axvline(self.best_epoch, linestyle="--", color="red")
            axs[1].scatter(self.best_epoch, min_y_val, edgecolor='black', marker='o', s=40)
        span_x = int(self.actual_epoch * 30 / 100)
        span_y = min_y_val * 300 / 100
        axs[1].set_xlim([self.best_epoch-span_x, self.best_epoch+span_x])
        axs[1].set_ylim([min_y_val-span_y, min_y_val+span_y*2])
        plt.legend()
        plt.suptitle(f"{self.model_name}: Training loss") 
        if save_path is not None: plt.savefig(f"{save_path}/history_{self.model_name}.pdf")
        plt.show()

    def predict(self, x, attempt=100, scaler:MMS=None, output_labels:str=None, returnDataFrame=False):
        means = []
        stds = []

        for data in x:
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32).to(self.device)
            
            predictions = np.zeros((attempt, len(output_labels)))

            for i in range(attempt):
                p = self(data).detach().cpu().numpy()
                if scaler is not None and output_labels is not None:
                    p = scaler.reverseArray(p, columns=output_labels)
                predictions[i] = p
            
            output_mean = np.mean(predictions, axis=0)
            output_stds = np.std(predictions, axis=0)

            means.append(output_mean)
            stds.append(output_stds)
        
        if returnDataFrame and output_labels is not None:
            return pd.DataFrame(means, columns=output_labels), pd.DataFrame(stds, columns=output_labels)
        else:
            return means, stds
    
    def _removeZeros(self, gt, pred=None, offset=1.5):
        if pred is None: pred = gt
        assert len(gt) == len(pred), "Array length mismatch"
        for i in range(len(gt)):
            gt[i] += offset; pred[i] += offset
        return gt, pred

    def _smape(self, gt: np.ndarray, pred: np.ndarray):
        abs_diff = np.abs(pred - gt)
        abs_sum = np.abs(pred) + np.abs(gt)
        gt_modified = gt.copy()
        gt_modified[gt_modified == 0] = abs_diff[gt_modified == 0] * 100
        return list((abs_diff / abs_sum) * 100)
    
    def _mae(self, gt: np.ndarray, pred: np.ndarray):
        return np.abs(gt - pred)

    def testModel(self, test_set:BNNDataset, scaler:MMS, output_labels:list, attempt = 10, skip = []):
        errors = []
        cov = []
        for i, data in enumerate(test_set):
            if i in skip: continue
            x,y = data
            y = scaler.reverseArray(y,columns=output_labels)
            preds, stds = self.predict([x], scaler=scaler, output_labels=output_labels, attempt=attempt)
            preds = np.array(preds[0])
            stds = stds[0]
            #y, preds = self._removeZeros(y, preds)
            errors.append(self._mae(y, preds))
            cov.append(stds)
        return errors, cov
    
    def save(self, path, subfolder = "AIModels/", save_model_resume = True):
        try: os.makedirs(subfolder)
        except FileExistsError as e: pass
        torch.save(self.state_dict(), path)
        if save_model_resume:
            self.__str__()
        print(f"{Color.green}[{Color.end}{self.model_name}{Color.green}] >> Model Saved{Color.end}")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        print(f"{Color.green}[{Color.end}{self.model_name}{Color.green}] >> Model Loaded on {Color.blue}{self.device}{Color.end}")


def test_multiple_models(models_to_test: List[BNN], test_set: BNNDataset, scaler: MMS, output_labels: list, attempt: int = 100, path: str = "test_results.csv", save_test_results: bool = True) -> pd.DataFrame:
    error_data = {"Model Name": []}
    for out_val in output_labels:
        error_data[out_val + " ERR%"] = []
        error_data[out_val + " STD%"] = []
        #error_data[out_val + " Max/Min Err %"] = []
        #error_data[out_val + " Max/Min CoV %"] = []
    error_data["ERR_TOT%"] = []
    for i, model in enumerate(models_to_test):
        error_data["Model Name"].append(model.model_name)
        errors, cov = model.testModel(test_set=test_set,
                                scaler=scaler,
                                output_labels=output_labels,
                                attempt=attempt)
        total_error = []
        for out_val in range(len(output_labels)):
            error_on_output_i = np.array([t[out_val] for t in errors])
            cov_on_output_i = np.array([t[out_val] for t in cov])
            cov_on_output_i = (cov_on_output_i / scaler.offset[output_labels[out_val]]) * 100
            error = np.mean(error_on_output_i) / scaler.offset[output_labels[out_val]] * 100
            total_error.append(error)
            error_data[output_labels[out_val] + " ERR%"].append(f"{error:.2f}")
            error_data[output_labels[out_val] + " STD%"].append(f"{np.mean(cov_on_output_i):.2f}")
            #error_data[output_labels[out_val] + " Max/Min Err %"].append(f"{np.max(error_on_output_i):.2f} / {np.min(error_on_output_i):.2f}")
            #error_data[output_labels[out_val] + " Max/Min CoV %"].append(f"{np.max(cov_on_output_i):.2f} / {np.min(cov_on_output_i):.2f}")
        error_data["ERR_TOT%"].append(np.sqrt(np.mean(np.array(total_error)**2)))
        
    error_table = pd.DataFrame(error_data)
    if save_test_results: error_table.to_csv(path, index=False)
    return error_table