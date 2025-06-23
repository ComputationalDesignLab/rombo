import torch 
import numpy as np
from smt.sampling_methods import LHS
from rombo.dimensionality_reduction.autoencoder import MLPAutoEnc
from rombo.rom.nonlinrom import AUTOENCROM
from rombo.test_problems.test_problems import LangermannFunction
from botorch.models import KroneckerMultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

# Setting data type and device for Pytorch based libraries
tkwargs = {
    "dtype": torch.float64,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

# Defining environment model function 
problem = LangermannFunction(input_dim = 16, output_dim = 60, normalized = False)

# Creating the training data
n_data = 10
xlimits = np.array([[0.0, 10.0]]*problem.input_dim)
sampler = LHS(xlimits=xlimits, criterion="ese")
xtrain = sampler(n_data)
xtrain = torch.tensor(xtrain, **tkwargs)
htrain = problem.evaluate(xtrain).flatten(1)
a = torch.rand(5,1,16)
h = problem.evaluate(a)
print(problem.utility(h).shape)
# print(xtrain)
# print(problem.utility(htrain))

# Generating the test data
test_sampler = LHS(xlimits=xlimits, criterion="ese")
xtest = test_sampler(10)
xtest = torch.tensor(xtest, **tkwargs)
htest = problem.evaluate(xtest).flatten(1)

# # Generating the nonlinear ROM model
# autoencoder = MLPAutoEnc(high_dim=problem.output_dim, hidden_dims=[256,64], zd = 10, activation = torch.nn.SiLU()).double()
# rom = AUTOENCROM(xtrain, htrain, autoencoder = autoencoder, low_dim_model = KroneckerMultiTaskGP, low_dim_likelihood = ExactMarginalLogLikelihood)

# # Training the ROM and predicting on the test data
# rom.trainROM(verbose=False)
# field = rom.predictROM(xtest)

# print(field[0] - htest[0])
# print(np.ptp(htest[0]))


