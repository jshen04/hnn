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

import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

def get_dataset(folder, speed, scaled=False, split=250000):
    normal_data = pd.read_csv('./Data to UNCC/{}/{}.csv'.format(folder, speed))
    normal_data = normal_data.to_numpy()
    if scaled:
        scaler = StandardScaler()
        data_fit = scaler.fit_transform(normal_data)
        return data_fit[:split]
    else:
        return normal_data[:split]

def get_dataset_split(folder, speed, scaled=False):
    data=get_dataset(folder, speed, scaled, split=250000)
    v, a1, a2, sound = data[:,0].reshape(-1,1), data[:,1:4], data[:,4:7], data[:,7].reshape(-1,1)
    return v, a1, a2, sound



### FOR DYNAMICS IN ANALYSIS SECTION ###
def hamiltonian_fn(coords):
  q, p = np.split(coords,2)
  H = p**2 + q# pendulum hamiltonian
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

def get_field(xmin=-30, xmax=10, ymin=-1, ymax=5, gridsize=40):
    field = {'meta': locals()}

    # meshgrid to get vector field
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    ys = np.stack([b.flatten(), a.flatten()])

    # get vector directions
    dydt = [dynamics_fn(None, y) for y in ys.T]
    dydt = np.stack(dydt).T

    field['x'] = ys.T
    field['dx'] = dydt.T
    return field