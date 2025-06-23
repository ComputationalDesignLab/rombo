import torch 
import numpy as np
from smt.sampling_methods import LHS
from rombo.dimensionality_reduction.autoencoder import MLPAutoEnc
from rombo.rom.nonlinrom import AUTOENCROM
from rombo.test_problems.test_problems import RosenbrockFunction
from botorch.models import KroneckerMultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

# Setting data type and device for Pytorch based libraries
tkwargs = {
    "dtype": torch.float64,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

# Defining environment model function 
problem = RosenbrockFunction(input_dim = 14, output_dim = 26, normalized = False)

# Creating the training data
n_data = 5
xlimits = np.array([[-4.0, 4.0]]*problem.input_dim)
sampler = LHS(xlimits=xlimits, criterion="ese", random_state=1)
xtrain = sampler(n_data)
xtrain = torch.tensor(xtrain, **tkwargs)
htrain = problem.evaluate(xtrain).flatten(1)
print(htrain.shape)
print(xtrain)
#print(htrain)
#print(xtrain)
ytrain = problem.utility(htrain)
print(ytrain)

a = torch.tensor([[1.0]*14])
h = problem.evaluate(a).flatten(1)
print(problem.utility(h))

# Generating the test data
test_sampler = LHS(xlimits=xlimits, criterion="ese", random_state=1)
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


