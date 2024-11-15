import numpy as np
import torch
from smt.sampling_methods import LHS
from rombo.test_problems.test_problems import EnvModelFunction
from rombo.dimensionality_reduction.autoencoder import ConditionalAutoEnc
from rombo.dimensionality_reduction.dim_red import AutoencoderReduction

tkwargs = {"device": torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0"), "dtype": torch.float}

# Defining the problem
problem = EnvModelFunction(input_dim=4, output_dim=64, normalized=False)

# Generating the training design of experiments
xlimits = np.array([[7.0, 13.0], [0.02, 0.12], [1.0, 3.0], [30.010, 30.295]])
train_sampler = LHS(xlimits=xlimits, criterion="ese")
xdoe = train_sampler(10)
xdoe[:,0] = np.rint(xdoe[:,0])
xdoe[:,2] = np.rint(xdoe[:,2])
xdoe = torch.tensor(xdoe, **tkwargs)
ydoe = problem.evaluate(xdoe).flatten(1)
xdoe_discrete = xdoe[:,[0,2]]
xdoe_cont = xdoe[:,[1,3]]
ydoe = torch.cat([ydoe, xdoe_discrete], dim = 1) 

# Generating the testing design of experiments
test_sampler = LHS(xlimits=xlimits, criterion="ese")
xtest = test_sampler(20)
xtest[:,0] = np.rint(xtest[:,0])
xtest[:,2] = np.rint(xtest[:,2])
xtest = torch.tensor(xtest)
ytest = problem.evaluate(xtest).flatten(1)
xtest_discrete = xtest[:,[0,2]]
xtest_cont = xtest[:,[1,3]]
ytest = torch.cat([ytest, xtest_discrete], dim = 1)

# Instantiating the model
model = ConditionalAutoEnc(high_dim = ydoe.shape[-1], hidden_dims = [128,64], activation = torch.nn.SiLU(), zd = 10, n_discrete = xdoe_discrete.shape[-1], x_discrete = xdoe_discrete)
dimred = AutoencoderReduction(S = ydoe, nn_model = model)
dimred.fit(epochs=1000)

