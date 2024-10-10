"Example script for building a ROM to predict the environment model function"

import torch 
import gpytorch
import math
from smt.sampling_methods import LHS
from aromad.rom.linrom import PODROM
from aromad.rom.nonlinrom import AUTOENCROM
import numpy as np
from botorch.models import KroneckerMultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from aromad.interpolation.models import MultitaskGPModel
from aromad.dimensionality_reduction.autoencoder import MLPAutoEnc
from aromad.test_problems.test_problems import EnvModelFunction
import warnings
warnings.filterwarnings('ignore')

tkwargs = {"device": torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0"), "dtype": torch.float}

# Defining environment model function
problem = EnvModelFunction(input_dim = 4, output_dim = 64)

# Creating the training data
n_data = 50
xlimits = np.array([[7.0,13.0], [0.02,0.12], [0.01,3.0], [30.010,30.295]])
sampler = LHS(xlimits=xlimits, criterion="ese")
xtrain = sampler(n_data)
xtrain = torch.tensor(xtrain, **tkwargs)
htrain = problem.evaluate(xtrain).flatten(1)

# Generating the test data
test_sampler = LHS(xlimits=xlimits, criterion="ese")
xtest = test_sampler(25)
xtest = torch.tensor(xtest, **tkwargs)
htest = problem.evaluate(xtest).flatten(1)

# Generating the ROM model
linrom = PODROM(xtrain, htrain, ric = 0.99, low_dim_model = MultitaskGPModel, low_dim_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood)
linrom.trainROM(verbose=False)
field = linrom.predictROM(xtest)

error_1 = []
for b in range(len(xtest)):
    
    error1 = torch.norm(field[b] - htest[b, :], p = 2) / torch.norm(htest[b, :], p = 2)
    error_1.append(error1.detach().cpu().numpy())

print("Mean Relative Error for Linear ROM:", np.mean(error_1))

# Generating the nonlinear ROM model
autoencoder = MLPAutoEnc(high_dim=problem.output_dim, hidden_dims=[128,64], zd = 10, activation = torch.nn.SiLU())
rom = AUTOENCROM(xtrain, htrain, autoencoder = autoencoder, low_dim_model = KroneckerMultiTaskGP, low_dim_likelihood = ExactMarginalLogLikelihood)
rom.trainROM(verbose=False)
field = rom.predictROM(xtest)

error_1 = []
for b in range(len(xtest)):
    
    error1 = torch.norm(field[b] - htest[b, :], p = 2) / torch.norm(htest[b, :], p = 2)
    error_1.append(error1.detach().cpu().numpy())

print("Mean Relative Error for Nonlinear ROM:", np.mean(error_1))

# Creating a plot of the true and predicted contours
x_plot = xtest[20].unsqueeze(0)
model_list = [rom, linrom]
color_list = ['r', 'b']
label_list = ['Autoencoder ROM', 'POD ROM']
problem.prediction_plotter(x_plot, model_list, color_list, label_list, save_filename='prediction.pdf')
