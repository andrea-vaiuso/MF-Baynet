# Multi-Fidelity Bayesian Neural Network (BNN) Library Documentation

## Overview

The Multi-Fidelity BNN library is designed to support the training, testing, and deployment of Bayesian Neural Networks (BNNs) with different fidelity levels. This library is particularly suited for scenarios where multiple levels of data fidelity are available, and the goal is to create models that can leverage this multi-fidelity data to make robust predictions with uncertainty estimates.

## Project Structure

The project is organized as follows:

```
AIModels/
Datasets/
Heatmaps/
Jupiter/
Settings/
  └── model_settings.yaml
  └── server_settings.yaml
  └── training_settings.yaml
bnn.py
cokriging.py
dataset_augmentation.ipynb
scalers.py
server.py
test.py
training.py
utils.py
```

### 1. `Settings/`

This directory contains YAML configuration files that define various settings used across the project.

- **`model_settings.yaml`**: Contains the hyperparameters and configurations for the Bayesian Neural Network models.
- **`server_settings.yaml`**: Defines the settings required to deploy the model on a server, including the host address, port number, and device information.
- **`training_settings.yaml`**: Specifies the training-related configurations, including training epochs, batch sizes, learning rates, and paths for saving models.

### 2. `bnn.py`

This module defines the Bayesian Neural Network (BNN) classes and methods for handling multi-fidelity data.

- **`BNNDataset`**: A custom dataset class that handles loading and managing multi-fidelity datasets. It includes methods for splitting the data into training, validation, and testing sets.
- **`BNN`**: The core Bayesian Neural Network class that handles the architecture, forward passes, training loops, and prediction routines. This class supports uncertainty estimation by performing multiple stochastic forward passes.
- **`test_multiple_models`**: A utility function to test multiple BNN models on a given dataset, useful for comparing the performance of models trained on different fidelity levels.

### 3. `cokriging.py`

This module defines the Co-Kriging (CK) model, which is used to combine predictions from low-fidelity and high-fidelity models.

- **`CK`**: The Co-Kriging class that integrates predictions from a low-fidelity BNN model with a Gaussian Process (GP) to improve the prediction accuracy for higher-fidelity data. This class includes methods for training the GP model, making predictions, and saving/loading the CK model.

### 4. `dataset_augmentation.ipynb`

This Jupyter Notebook is intended for augmenting datasets, possibly generating additional synthetic data points or performing preprocessing tasks before training the BNN models. The specifics of what this notebook accomplishes depend on its content, which might include data cleaning, feature engineering, or data transformation.

### 5. `scalers.py`

This module defines custom scalers used for normalizing data.

- **`MinMaxScaler`**: A custom implementation of the Min-Max scaler that scales the features to a specified range. This scaler is particularly useful in neural network training as it ensures that the input features are within a range that facilitates convergence.

### 6. `server.py`

This module provides the functionality to deploy a trained BNN model on a server. It allows the model to be used as a service, accepting input data over a network and returning predictions.

- **`run_server`**: A function that sets up and runs the server using the settings defined in `server_settings.yaml`.
- **`handle_request`**: A function to handle incoming requests to the server, process the input data, run the model, and return predictions.

### 7. `test.py`

This script is responsible for testing the trained BNN and CK models on a test dataset. It performs the following tasks:

- **Model Loading**: Loads the trained models from their saved state.
- **Dataset Loading**: Loads the test dataset and applies the necessary normalization.
- **Model Testing**: Runs the models on the test dataset, compares predictions against ground truth, and calculates error metrics.
- **Result Saving**: Saves the results of the model testing to a CSV file.

### 8. `training.py`

This script handles the end-to-end training process of the BNN models, including:

- **Settings Loading**: Reads all the configurations from the YAML files.
- **Data Preparation**: Loads and preprocesses the datasets, applies normalization, and splits them into training, validation, and testing sets.
- **Model Training**: Trains the low-fidelity, mid-fidelity, and fine-tuned models, as well as the Co-Kriging model.
- **Model Saving**: Saves the trained models and their configurations for later use.
- **Result Plotting**: Plots the predictions of the trained models and compares them with the validation data.

### 9. `utils.py`

This utility module provides helper functions that are used across various parts of the project.

- **`Color`**: A class that defines color codes for printing colored text in the console.
- **`show_continue_cancel`**: A function that prompts the user with a yes/no question to continue or cancel an operation.
- **`create_data_dump`**: A function that packages the model settings, dataset settings, training settings, and trained models into a data structure for saving as a YAML file.

## How to Use the Multi-Fidelity BNN Library

### Step 1: Configure the Settings
Before running the training or testing scripts, ensure that the YAML files in the `Settings/` directory are correctly configured. These files should define all necessary hyperparameters, file paths, and other configurations needed by the models and scripts.

### Step 2: Train the Models
Run the `training.py` script to start the training process. This script will:
- Load the dataset.
- Normalize the data.
- Train the low-fidelity, mid-fidelity, and transfer learning models.
- Train the Co-Kriging model using the low-fidelity model as a basis.
- Save all trained models and their configurations.

```bash
python training.py
```

### Step 3: Test the Models
After training, use the `test.py` script to evaluate the models on a test dataset. This script will:
- Load the trained models and test dataset.
- Evaluate the models and calculate performance metrics.
- Save the test results for analysis.

```bash
python test.py
```

### Step 4: Deploy the Model
Once the models are trained and tested, you can deploy them using the `server.py` script. This will set up a server that listens for incoming requests, runs the model on the input data, and returns the predictions.

```bash
python server.py
```

### Step 5: Augment and Prepare Data
If needed, you can use the `dataset_augmentation.ipynb` notebook to perform additional data augmentation before training. This is especially useful if your dataset is small or if you want to experiment with different data preprocessing techniques.

## Conclusion

The Multi-Fidelity BNN library is a powerful tool for training and deploying Bayesian Neural Networks in scenarios where multi-fidelity data is available. By carefully following the structure and steps outlined in this guide, you can efficiently develop robust models that not only make predictions but also provide uncertainty estimates, making them highly valuable in real-world applications where decision-making under uncertainty is critical.

For further questions or support, please refer to the documentation within each module or contact the development team.

---

This document should serve as a comprehensive guide to help you navigate the project and effectively utilize the Multi-Fidelity BNN library.
