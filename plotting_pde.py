from aromad.rom.nonlinrom import AUTOENCROM
import numpy as np
import torch
from botorch.models import KroneckerMultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from aromad.dimensionality_reduction.autoencoder import MLPAutoEnc
from scipy.io import loadmat
from aromad.test_problems.test_problems import BrusselatorPDE
import matplotlib.pyplot as plt

# Setting data type and device for Pytorch based libraries
tkwargs = {
    "dtype": torch.float,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

# Defining the objective function
objective = BrusselatorPDE(input_dim=32, Nx=64, Ny=64, tkwargs=tkwargs)

data = loadmat('brusselator_results_v1.mat')
bo_data = data['BO_EI']
bologei_data = data['BO_LOGEI']
romboei_data = data['ROMBO_EI']
rombologei_data = data['ROMBO_LOGEI']

# Plotting the objective function history
plt.rcParams['font.size'] = 12
fig, ax = plt.subplots()
iter = np.arange(20,50,1)
ax.plot(iter, -bo_data['objectives'][0][0][0], 'bo-', label = 'BO+EI')
ax.plot(iter, -bologei_data['objectives'][0][0][0], 'mo-', label = 'BO+Log EI')
ax.plot(iter, -romboei_data['objectives'][0][0][0], 'ro-', label = 'ROMBO+EI')
ax.plot(iter, -rombologei_data['objectives'][0][0][0], 'go-', label = 'ROMBO+Log EI')
ax.grid()
ax.legend(loc = 'upper right')
ax.set_xlabel('Number of Samples')
ax.set_ylabel('Best Objective Function Value, $y^*$')
plt.tight_layout()
plt.savefig('./brusselatorpde_objective.pdf')
plt.show()

# Plotting the contours of the function
x_list = [bo_data['design'][0][0][0][-1], bologei_data['design'][0][0][0][-1], romboei_data['design'][0][0][0][-1], rombologei_data['design'][0][0][0][-1]]
x_list = [bo_data["xdoe"][0][0][int(x_list[0])], bologei_data["xdoe"][0][0][int(x_list[1])], romboei_data["xdoe"][0][0][int(x_list[2])], rombologei_data["xdoe"][0][0][int(x_list[3])]]
x_list = torch.tensor(x_list, **tkwargs)
autoencoder = MLPAutoEnc(high_dim=objective.output_dim, hidden_dims=[128,64,32], zd = 10, activation = torch.nn.SiLU())
rom = AUTOENCROM(romboei_data['xdoe'][0][0], romboei_data['ydoe'][0][0], autoencoder = autoencoder, low_dim_model = KroneckerMultiTaskGP, low_dim_likelihood = ExactMarginalLogLikelihood)
rom.trainROM(verbose=False)

model_list = [rom]
color_list = ['r']
label_list = ['Autoencoder ROM']
objective.prediction_plotter(x_list[2].unsqueeze(0), model_list, color_list, label_list, plot_true=False)
