# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch
import numpy as np
import pandas as pd
import scipy, scipy.misc
import numpy
from sklearn.preprocessing import StandardScaler

import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

def get_measurments(folder):
    normal_data = pd.read_csv('./Data to UNCC/{}/14.csv'.format(folder))
    normal_data = normal_data.to_numpy()
    return normal_data

def get_dataset_clean(folder):
    normal_data = pd.read_csv('./Data to UNCC/{}/14.csv'.format(folder))
    normal_data = normal_data.to_numpy()
    scaler = StandardScaler()
    data_fit=scaler.fit_transform(normal_data)
    return data_fit

### FOR DYNAMICS IN ANALYSIS SECTION ###
def hamiltonian_fn(coords):
  q, p = np.split(coords,2,axis=1)
  H = p**2 + q # pendulum hamiltonian
  return H

def hamiltonian_fn_torch(coords):
  q, p = torch.split(coords, int(coords.shape[1] / 2), 1)
  H = p**2 + q # pendulum hamiltonian
  return H

def dynamics_fn(t, coords):
  dcoords = autograd.grad(hamiltonian_fn)(coords)
  dqdt, dpdt = np.split(dcoords,2)
  S = -np.concatenate([dpdt, -dqdt], axis=-1)
  return S

