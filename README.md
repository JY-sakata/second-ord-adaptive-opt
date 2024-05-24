# adaptive-second-ord-opt


Does Using Second-Order Information for Step Size Adaptation Improve Neural Network Convergence Speed?

## Overview

This project focuses on the analysis and visualisation of different optimizers' performance in training neural networks. It includes various scripts and Jupyter notebooks for data analysis, visualisation, and training neural networks with different configurations and optimizers.

## Directory Structure

- **datasets/**: Directory for storing datasets.
- **models/**: Directory for storing trained models.
- **optimizers/**: Directory for storing different optimisers implementation.
- **README.md**: Project overview and instructions.
- **RR_data_visualisation.ipynb**: Jupyter notebook for visualising optimisers performance on the Rahimi-Recht function.
- **neural_network_task_data_visualisation.ipynb**: Jupyter notebook for visualising optimisers performance on the two network learning tasks.
- **data_visualisation_utils.py**: Utility functions for data visualisation.
- **hessian_analysis.py**: Script for performing Hessian analysis used in the Ramihi-Recht function.
- **optimiser_configuration.py**: Script for optimisers configuration.
- **requirements.txt**: List of dependencies required to run the project.
- **train_resnet18_by_iterations.py**: Script for training ResNet18 with iteration-based analysis.
- **net_RR_cond15099_xdim12_wdim13_ydim14_nlayers2.pth**: Rahimi-Recht function with a condition number of 15099.
- **net_RR_cond29_xdim12_wdim13_ydim14_nlayers2.pth**: Rahimi-Recht function with a condition number of 29.
- **train_transformer.py**: Script for training transformer models.
- **tune_and_train_RR.py**: Script for tuning hyperparameters for optimisers and training on the Rahimi-Recht function.
- **tune_resnet18.py**: Script for tuning hyperparameters for optimisers in training the ResNet18.
- **tune_transformer.py**: Script for tuning hyperparameters for optimiser in training the transformer.

## Getting Started

### Prerequisites

Ensure you have the following dependencies installed:

- Python 3.8+
- Jupyter Notebook
- Matplotlib
- NumPy
- PyTorch
- Ray
- Other dependencies listed in `requirements.txt`


#### Training Neural Networks

- Use `tune_and_train_RR.py`, `train_resnet18_by_iterations.py`, and `train_transformer.py` to train neural network on three tasks.
- The scripts `tune_and_train_RR.py`, `tune_resnet18.py`, and `tune_transformer.py` are for tuning hyperparameters of the respective models.

#### Optimiser Configuration

- Find best optimiser configuration for three tasks in `optimiser_configuration.py` .

#### Data Visualiasation

- ResNet18: https://wandb.ai/1848014170/train_resnet18?nw=nwuser1848014170
- Transformer: https://wandb.ai/1848014170/train_transformer?nw=nwuser1848014170
- Rahimi-Recht Functions: Reproduce experiments in `tune_and_train_RR.py` (Due to the large file sizes, previous results for Rahimi-Recht function fail to upload)




## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Special thanks to the contributors and the open-source community.
- References and inspiration from various machine learning and deep learning resources.
