# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch
import autograd
import numpy as np
import pandas as pd
import scipy, scipy.misc
import numpy
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import os, sys, glob
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

def get_dataset(folder, speed, scaled=False, split=250000, tensor=False, experiment_dir='.'):
    path = '{}/dataset/{}/{}*.csv'.format(experiment_dir, folder, speed)
    print(path)
    path=glob.glob(path)[0]
    normal_data = pd.read_csv(path)
    normal_data = normal_data.to_numpy()
    if scaled:
        scaler = StandardScaler()
        data_fit = scaler.fit_transform(normal_data)
        if tensor:
            return torch.tensor(data_fit[:split], dtype=torch.float)
        else:
            return data_fit[:split]
    else:
        if tensor:
            return torch.tensor(normal_data[:split], dtype=torch.float)
        else:
            return normal_data[:split]

def get_dataset_split(folder, speed, scaled=False, tensor=False, experiment_dir='.'):
    data=get_dataset(folder, speed, scaled, split=250000, tensor=tensor, experiment_dir=experiment_dir)
    v, a1, a2, sound = data[:,0].reshape(-1,1), data[:,1:4], data[:,4:7], data[:,7].reshape(-1,1)
    return v, a1, a2, sound

def get_dataset_range(folder='normal', speeds=[14, 15, 20, 25, 30, 35, 40, 61], experiment_dir='.'):
    data=np.zeros((1,8))
    for speed in speeds:
        file=get_dataset(folder, speed, experiment_dir=experiment_dir)
        data=np.vstack((data, file))
    return data[1:]