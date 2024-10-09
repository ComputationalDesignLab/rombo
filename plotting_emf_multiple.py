from aromad.rom.nonlinrom import AUTOENCROM
import numpy as np
from botorch.models import KroneckerMultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from aromad.dimensionality_reduction.autoencoder import MLPAutoEnc
import torch
from scipy.io import loadmat
from aromad.test_problems.test_problems import EnvModelFunction
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Setting data type and device for Pytorch based libraries
tkwargs = {
    "dtype": torch.float,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

objective = EnvModelFunction(input_dim=15, output_dim=256, normalized=True)
data = loadmat('env_model_results_256_multiples_trials.mat')
bo_data = data['BO_EI']
bologei_data = data['BO_LOGEI']
romboei_data = data['ROMBO_EI']
rombologei_data = data['ROMBO_LOGEI']

data_list = [bo_data['objectives'][0][0], bologei_data['objectives'][0][0], romboei_data['objectives'][0][0], rombologei_data['objectives'][0][0]]

# Plotting the objective function history
plt.rcParams['font.size'] = 12
fig, ax = plt.subplots()
iter = np.arange(10,40,1)
color_list = ['bo-','mo-','ro-','go-']
labels = ['BO+EI','BO+Log EI','ROMBO+EI','ROMBO+Log EI']
fill_color = ['cornflowerblue','magenta','lightcoral','lightgreen']
i = 0
for data in data_list:
    
    # Calculating statistics of objectives and design variables for optimization
    objectives = -data
    mean_objectives = objectives.mean(axis = 0)
    objectives_std = objectives.std(axis = 0)

    lower_objectives = mean_objectives - objectives_std
    upper_objectives = mean_objectives + objectives_std
    
    lower_objectives[lower_objectives < 1e-3] = 1e-2
    
    plt.semilogy(iter, mean_objectives, color_list[i], label = labels[i])
    plt.fill_between(iter, lower_objectives, upper_objectives, color = fill_color[i], alpha=0.3, zorder=100)
    i+=1

ax.grid()
ax.legend()
ax.set_xlabel('Number of Samples')
ax.set_ylabel('Best Objective Function Value, $y^*$')
plt.tight_layout()
plt.show()

# Plotting the contours of the function
# x_list = [bo_data['design'][0][0][0][-1][0], bologei_data['design'][0][0][0][-1], romboei_data['design'][0][0][0][-1][0], rombologei_data['design'][0][0][0][-1][0]]
# x_list = [bo_data["xdoe"][0][0][int(x_list[0])], bologei_data["xdoe"][0][0][int(x_list[1])], romboei_data["xdoe"][0][0][int(x_list[2])], rombologei_data["xdoe"][0][0][int(x_list[3])]]
# x_list = torch.tensor(x_list, **tkwargs)
# color_list = ['b', 'm', 'r', 'g']
# label_list = ['BO + EI', 'BO + Log EI', 'ROMBO + EI', 'ROMBO + Log EI']
# objective.optresult_plotter(x_list, color_list, label_list)

# # Generating the nonlinear ROM model
# autoencoder = MLPAutoEnc(high_dim=objective.output_dim, hidden_dims=[128,64], zd = 10, activation = torch.nn.SiLU())
# rom = AUTOENCROM(romboei_data['xdoe'][0][0], romboei_data['ydoe'][0][0], autoencoder = autoencoder, low_dim_model = KroneckerMultiTaskGP, low_dim_likelihood = ExactMarginalLogLikelihood)
# rom.trainROM(verbose=False)

# model_list = [rom]
# color_list = ['r']
# label_list = ['Autoencoder ROM']
# objective.prediction_plotter(x_list[2].unsqueeze(0), model_list, color_list, label_list)

