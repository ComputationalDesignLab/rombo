from rombo.rom.nonlinrom import AUTOENCROM
import numpy as np
from botorch.models import KroneckerMultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from rombo.dimensionality_reduction.autoencoder import MLPAutoEnc
import torch
from scipy.io import loadmat
from rombo.test_problems.test_problems import EnvModelFunction
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Setting data type and device for Pytorch based libraries
tkwargs = {
    "dtype": torch.float,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

n_trials = 20
n_init = 10
n_iterations = 40
objective = EnvModelFunction(input_dim=64, output_dim=1024, normalized=True)

boei_objectives = np.zeros((n_trials, n_iterations))
bologei_objectives = np.zeros((n_trials, n_iterations))
romboei_objectives = np.zeros((n_trials, n_iterations))
rombologei_objectives = np.zeros((n_trials, n_iterations))

boei_dvs = np.zeros((n_trials, n_iterations))
bologei_dvs = np.zeros((n_trials, n_iterations))
romboei_dvs = np.zeros((n_trials, n_iterations))
rombologei_dvs = np.zeros((n_trials, n_iterations))

boei_t = np.zeros((n_trials, n_iterations))
bologei_t = np.zeros((n_trials, n_iterations))
romboei_t = np.zeros((n_trials, n_iterations))
rombologei_t = np.zeros((n_trials, n_iterations))

boei_doe = np.zeros((n_trials, n_iterations+n_init, objective.input_dim))
bologei_doe = np.zeros((n_trials, n_iterations+n_init, objective.input_dim))
romboei_doe = np.zeros((n_trials, n_iterations+n_init, objective.input_dim))
rombologei_doe = np.zeros((n_trials, n_iterations+n_init, objective.input_dim))

for trial in range(20):

    data = loadmat('./EMF/emf_{}_512/env_model_results_{}_1024_BO_trial_{}.mat'.format(objective.input_dim,objective.input_dim,trial+1))
    bo_data = data['BO_EI']
    bologei_data = data['BO_LOGEI']
    romboei_data = data['ROMBO_EI']
    rombologei_data = data['ROMBO_LOGEI']

    boei_objectives[trial] = bo_data['objectives'][0][0]
    bologei_objectives[trial] = bologei_data['objectives'][0][0]
    romboei_objectives[trial] = romboei_data['objectives'][0][0]
    rombologei_objectives[trial] = rombologei_data['objectives'][0][0]

    boei_dvs[trial] = bo_data['design'][0][0]
    bologei_dvs[trial] = bologei_data['design'][0][0]
    romboei_dvs[trial] = romboei_data['design'][0][0]
    rombologei_dvs[trial] = rombologei_data['design'][0][0]

    boei_doe[trial] = bo_data['doe'][0][0][0]
    bologei_doe[trial] = bologei_data['doe'][0][0][0]
    romboei_doe[trial] = romboei_data['doe'][0][0][0]
    rombologei_doe[trial] = rombologei_data['doe'][0][0][0]

    boei_t[trial] = bo_data['time'][0][0]
    bologei_t[trial] = bologei_data['time'][0][0]
    romboei_t[trial] = romboei_data['time'][0][0]
    rombologei_t[trial] = rombologei_data['time'][0][0]
        
#data_list = [boei_objectives, bologei_objectives, romboei_objectives, rombologei_objectives]
data_list = [boei_t, bologei_t, romboei_t, rombologei_t]
print(bologei_t)
# Plotting the objective function history
plt.rcParams['font.size'] = 12
fig, ax = plt.subplots()
iter = np.arange(1,41,1)
color_list = ['b-','m-','r-','g-']
labels = ['BO+EI','BO+Log EI','ROMBO+EI','ROMBO+Log EI']
fill_color = ['cornflowerblue','magenta','lightcoral','lightgreen']
i = 0
for data in data_list:
    
    # Calculating statistics of objectives and design variables for optimization
    objectives = data
    mean_objectives = objectives.mean(axis = 0)
    objectives_std = objectives.std(axis = 0)

    lower_objectives = mean_objectives - objectives_std
    upper_objectives = mean_objectives + objectives_std
    
    #lower_objectives[lower_objectives < 1e-3] = 1e-1
    
    plt.plot(iter, mean_objectives, color_list[i], label = labels[i], linewidth=3.5)
    #plt.fill_between(iter, lower_objectives, upper_objectives, color = fill_color[i], alpha=0.3, zorder=100)
    i+=1

ax.grid()
ax.legend()
ax.set_xlabel('Iteration')
#ax.set_ylabel('Log(Best Objective Function Value), $\log_{10}(y^*)$')
ax.set_ylabel('Average Time (s)')
ax.set_xlim([1,40])
plt.tight_layout()
plt.savefig('emf_512_64_time.pdf')
#plt.show()

# Finding the best trial and corresponding design variables
trial_idx = [boei_objectives[:,-1].argmax(), bologei_objectives[:,-1].argmax(),
             romboei_objectives[:,-1].argmax(), rombologei_objectives[:,-1].argmax()]

# Plotting the contours of the function
x_list = [boei_dvs[trial_idx[0]][-1], bologei_dvs[trial_idx[1]][-1], romboei_dvs[trial_idx[2]][-1], rombologei_dvs[trial_idx[3]][-1]]
x_list = [boei_doe[trial_idx[0]][int(x_list[0])], bologei_doe[trial_idx[1]][int(x_list[1])], romboei_doe[trial_idx[2]][int(x_list[2])], rombologei_doe[trial_idx[3]][int(x_list[3])]]
x_list = torch.tensor(x_list, **tkwargs)
color_list = ['b', 'm', 'r', 'g']
label_list = ['BO + EI', 'BO + Log EI', 'ROMBO + EI', 'ROMBO + Log EI']
linestyle_list = ['-', '-', '-', '-']
objective.optresult_plotter(x_list, color_list, label_list, linestyle_list, filename='emf_512_64_prediction.pdf')

# # Generating the nonlinear ROM model
# autoencoder = MLPAutoEnc(high_dim=objective.output_dim, hidden_dims=[128,64], zd = 10, activation = torch.nn.SiLU())
# rom = AUTOENCROM(romboei_data['xdoe'][0][0], romboei_data['ydoe'][0][0], autoencoder = autoencoder, low_dim_model = KroneckerMultiTaskGP, low_dim_likelihood = ExactMarginalLogLikelihood)
# rom.trainROM(verbose=False)

# model_list = [rom]
# color_list = ['r']
# label_list = ['Autoencoder ROM']
# objective.prediction_plotter(x_list[2].unsqueeze(0), model_list, color_list, label_list)

