import numpy as np
import torch
from scipy.io import savemat
from aromad.test_problems.test_problems import BrusselatorPDE
from smt.sampling_methods import LHS

tkwargs = {"device": torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0"), "dtype": torch.float}

# Defining the problem
problem = BrusselatorPDE(input_dim=4, Nx=64, Ny=64, tkwargs=tkwargs)

# Generating the design of experiments
n_data = 20
xlimits = np.array([[0.0,1.0]]*32)
sampler = LHS(xlimits=xlimits, criterion="ese")
xtrain = sampler(n_data)
ytrain = problem.evaluate(xtrain)

# Saving the samples generated
samples = {"x": torch.tensor(xtrain, **tkwargs), "y": ytrain}
savemat("bruss_init.mat", samples)
