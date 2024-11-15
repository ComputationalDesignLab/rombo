import numpy as np
import torch
from scipy.io import savemat
from rombo.test_problems.test_problems import BrusselatorPDE
from smt.sampling_methods import LHS

tkwargs = {"device": torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0"), "dtype": torch.float}

# Defining the problem
input_dim = 128
problem = BrusselatorPDE(input_dim=input_dim, Nx=64, Ny=64, tkwargs=tkwargs)

# Generating data for multiple trials of the Brusselator PDE method

# Generating the design of experiments
n_data = 15
n_trials = 20
xtrain = np.zeros((n_trials, n_data, input_dim))
ytrain = torch.zeros((n_trials, n_data, 2, problem.Nx, problem.Ny))
for trial in range(n_trials):
    print("\n\n### Running trial {} ###".format(trial+1))
    xlimits = np.array([[0.0,1.0]]*input_dim)
    sampler = LHS(xlimits=xlimits, criterion="ese")
    X = sampler(n_data)
    xtrain[trial,:,:] = X
    ytrain[trial,:,:,:,:] = problem.evaluate(X)

# Saving the samples generated
samples = {"x": torch.tensor(xtrain, **tkwargs), "y": ytrain}
savemat("bruss_data_multrial.mat", samples)
