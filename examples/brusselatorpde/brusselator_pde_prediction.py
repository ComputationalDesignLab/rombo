"Example script for building a ROM to predict the environment model function"

import torch 
import gpytorch
import math
from smt.sampling_methods import LHS
from aromad.rom.linrom import PODROM
from aromad.rom.nonlinrom import AUTOENCROM
import numpy as np
from scipy.io import loadmat
from botorch.models import KroneckerMultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from aromad.interpolation.models import MultitaskGPModel
from aromad.dimensionality_reduction.autoencoder import MLPAutoEnc
from aromad.test_problems.test_problems import BrusselatorPDE
from aromad.dimensionality_reduction.dim_red import AutoencoderReduction

tkwargs = {"device": torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0"), "dtype": torch.float}

# Defining environment model function
problem = BrusselatorPDE(input_dim=32, Nx=64, Ny=64, tkwargs=tkwargs)

# Creating the training design of experiments
data = loadmat('./bruss_data_50.mat')
bounds = torch.cat((torch.zeros(1, 32), torch.ones(1, 32))).to(**tkwargs)

xtrain = data['x']
htrain = data['y']
htrain = htrain.reshape((htrain.shape[0], problem.output_dim))
htrain = torch.tensor(htrain, **tkwargs)

# Generating the test data
test_data = loadmat('./bruss_init.mat')
xtest = test_data['x']
xtest = torch.tensor(xtest, **tkwargs)
htest = test_data['y']
htest = htest.reshape((htest.shape[0], problem.output_dim))
htest = torch.tensor(htest, **tkwargs)

# # Generating the nonlinear ROM model
autoencoder = MLPAutoEnc(high_dim=problem.output_dim, hidden_dims=[256,128,64], zd = 10, activation = torch.nn.SiLU())
rom = AUTOENCROM(xtrain, htrain, autoencoder = autoencoder, low_dim_model = KroneckerMultiTaskGP, low_dim_likelihood = ExactMarginalLogLikelihood, standard = False)
rom.trainROM(verbose=True)
field = rom.predictROM(xtest)

a_true = rom.dimensionreduction.model.encoder(htest)
predicted_a, _ = rom.gp_model.predict(xtest)
field = rom.dimensionreduction.backmapping(predicted_a)

error_1 = []
for b in range(len(xtest)):
    
    error1 = torch.norm(field[b] - htest[b, :], p = 2) / torch.norm(htest[b, :], p = 2)
    error_1.append(error1.detach().cpu().numpy())

print("Mean Relative Error for Nonlinear ROM:", np.mean(error_1))

# Creating a plot of the true and predicted contours
# x_plot = xtest[20].unsqueeze(0)
# model_list = [rom, linrom]
# color_list = ['b', 'r']
# label_list = ['Autoencoder ROM', 'POD ROM']
# problem.prediction_plotter(x_plot, model_list, color_list, label_list)
