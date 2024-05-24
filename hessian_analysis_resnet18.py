import numpy as np
import torch 
from torchvision import datasets, transforms
from pyhessian import hessian # Hessian computation
from pyhessian.utils import *
from pyhessian import hessian
from models import ResNet18
import matplotlib.pyplot as plt
from datasets import get_cifar10_dataloader
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import dataset




def get_hessian_analysis(net,loss_func, partial_dataloader, ):
    net.eval()
    hessian_computation = hessian(net, loss_func, data=None, dataloader=partial_dataloader)
    eigenvals, eigenvecs = hessian_computation.eigenvalues(top_n=2)
    trace_vector = hessian_computation.trace()
    density_eigen, density_weight = hessian_computation.density()
    metrics = {
        'net_state': net.state_dict(),
        'eigenvalues': eigenvals,
        'eigenvectors': eigenvecs,
        'trace_vector': trace_vector,
        'density_eigen': density_eigen,
        'density_weight': density_weight,
    }
    return metrics
