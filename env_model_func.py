"Example script for building a ROM to predict the environment model function"

import torch 
import gpytorch
import math
from smt.sampling_methods import LHS
from aromad.rom.linrom import PODROM
from aromad.rom.nonlinrom import AUTOENCROM
import numpy as np
from aromad.interpolation.models import MultitaskGPModel
from aromad.dimensionality_reduction.autoencoder import MLPAutoEnc

tkwargs = {"device": torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0"), "dtype": torch.float}

# Defining environment model function
def env_cfun(M, D, L, tau, s, t):
    c1 = M / torch.sqrt(4 * math.pi * D * t)
    exp1 = torch.exp(-(s ** 2) / 4 / D / t)
    term1 = c1 * exp1
    c2 = M / torch.sqrt(4 * math.pi * D * (t - tau))
    exp2 = torch.exp(-((s - L) ** 2) / 4 / D / (t - tau))
    term2 = c2 * exp2
    term2[torch.isnan(term2)] = 0.0
    return term1 + term2

def c_batched(X, Sgrid, Tgrid):
    return torch.stack([env_cfun(*x, Sgrid, Tgrid) for x in X])

# Defining the grid parametes
s_size = 10
t_size = 10
n_data = 100

# Defining spatial and temporal grid
S = torch.linspace(0.0, 3.0, s_size, **tkwargs)
T = torch.linspace(15.0, 60.0, t_size, **tkwargs)
Sgrid, Tgrid = torch.meshgrid(S, T)

xlimits = np.array([[7.0,13.0], [0.02,0.12], [0.01,3.0], [30.010,30.295]])

# Creating the training data
sampler = LHS(xlimits=xlimits, criterion="ese")
xtrain = sampler(n_data)
xtrain = torch.tensor(xtrain, **tkwargs)
htrain = c_batched(xtrain, Sgrid, Tgrid).flatten(1)

# Generating the test data
test_sampler = LHS(xlimits=xlimits, criterion="ese")
xtest = test_sampler(25)
xtest = torch.tensor(xtest, **tkwargs)
htest = c_batched(xtest, Sgrid, Tgrid).flatten(1)

# Generating the ROM model
rom = PODROM(xtrain, htrain, ric = 0.99, low_dim_model = MultitaskGPModel, low_dim_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood)
rom.trainROM(verbose=False)
field = rom.predictROM(xtest)

error_1 = []
for b in range(len(xtest)):
    
    error1 = torch.norm(field[b] - htest[b, :], p = 2) / torch.norm(htest[b, :], p = 2)
    error_1.append(error1.detach().cpu().numpy())

print("Mean Relative Error for Linear ROM:", np.mean(error_1))

# Generating the nonlinear ROM model
autoencoder = MLPAutoEnc(high_dim=s_size*t_size, hidden_dims=[128,64], zd = 10, activation = torch.nn.SiLU())
rom = AUTOENCROM(xtrain, htrain, autoencoder = autoencoder, low_dim_model = MultitaskGPModel, low_dim_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood)
rom.trainROM(verbose=False)
field = rom.predictROM(xtest)

error_1 = []
for b in range(len(xtest)):
    
    error1 = torch.norm(field[b] - htest[b, :], p = 2) / torch.norm(htest[b, :], p = 2)
    error_1.append(error1.detach().cpu().numpy())

print("Mean Relative Error for Nonlinear ROM:", np.mean(error_1))

