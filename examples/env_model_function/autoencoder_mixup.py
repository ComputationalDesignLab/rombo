"Using MixUp to enhance global accuracy of autoencoders"

import torch 
import gpytorch
import math
from smt.sampling_methods import LHS
from aromad.dimensionality_reduction.dim_red import AutoencoderReduction
import numpy as np
from aromad.dimensionality_reduction.autoencoder import MLPAutoEnc
from aromad.test_problems.test_problems import EnvModelFunction
import warnings
warnings.filterwarnings('ignore')

tkwargs = {"device": torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0"), "dtype": torch.float}

def mixup_data(x, alpha=1.0):
    
    batch_size = x.size(0)

    # Sample lambda from the beta distribution
    if alpha > 0:
        lam = np.random.exponential(alpha)
    else:
        lam = 1

    # Randomly select two indices
    index1 = torch.randperm(batch_size)
    index2 = torch.randperm(batch_size)

    # Create mixed inputs and targets
    mixed_x = lam * x[index2, :] + (1 - lam) * x[index1, :]

    return mixed_x

# Defining environment model function
problem = EnvModelFunction(input_dim = 15, output_dim = 64, normalized = True)

# Creating the training data
n_data = 10
xlimits = np.array([[0.0,1.0]]*15)
sampler = LHS(xlimits=xlimits, criterion="ese")
xtrain = sampler(n_data)
xtrain = torch.tensor(xtrain, **tkwargs)
htrain = problem.evaluate(xtrain)
htrain = htrain.reshape((htrain.shape[0], problem.output_dim))

# MixUp iterations to enhance the data
mixed_h = mixup_data(htrain, alpha=2)
for i in range(9):
    mixed_h = torch.cat([mixed_h, mixup_data(htrain, alpha=2)])
htrain = torch.cat([htrain, mixed_h], dim = 0)

# Calculating initial scores for standard BO procedure
score_doe = problem.utility(htrain)

# Generating the test data
test_sampler = LHS(xlimits=xlimits, criterion="ese")
xtest = test_sampler(50)
xtest = torch.tensor(xtest, **tkwargs)
htest = problem.evaluate(xtest)
htest = htest.reshape((htest.shape[0], problem.output_dim))

autoencoder = MLPAutoEnc(high_dim=problem.output_dim, hidden_dims=[128,64], zd = 10, activation = torch.nn.SiLU())
dimension_reduce = AutoencoderReduction(htrain, nn_model = autoencoder)
dimension_reduce.fit(epochs=1000)

field = dimension_reduce.model(htest)

error_1 = []
for b in range(len(xtest)):
    
    error1 = torch.norm(field[b] - htest[b, :], p = 2) / torch.norm(htest[b, :], p = 2)
    error_1.append(error1.detach().cpu().numpy())

print("Mean Relative Error for autoencoder:", np.mean(error_1))