"Example script for building multiple ROM optimizers, solving the environment model problem and comparing the results"

# Importing standard libraries
import torch 
from smt.sampling_methods import LHS
from aromad.rom.nonlinrom import AUTOENCROM
import numpy as np
from aromad.dimensionality_reduction.autoencoder import MLPAutoEnc
from aromad.test_problems.test_problems import EnvModelFunction
from aromad.optimization.rombo import ROMBO
from aromad.optimization.altbo import BO
from scipy.io import savemat, loadmat
import warnings
warnings.filterwarnings('ignore')

# Importing relevant classes from BoTorch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.models import KroneckerMultiTaskGP, SingleTaskGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

tkwargs = {"device": torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0"), "dtype": torch.float}

# Creating the initial design of experiments
#xlimits = np.array([[7.0, 13.0], [0.02, 0.12], [0.01, 3.0], [30.010, 30.295]])
xlimits = np.array([[0.0, 1.0]]*15)
sampler = LHS(xlimits=xlimits, criterion="ese")
n_init = 10
objective = EnvModelFunction(input_dim=15, output_dim=256, normalized=True)
#bounds = torch.tensor([[7.0, 0.02, 0.01, 30.010], [13.0, 0.12, 3.00, 30.295]], **tkwargs)
bounds = torch.cat((torch.zeros(1, 15), torch.ones(1, 15))).to(**tkwargs)

xdoe = sampler(n_init)
xdoe = torch.tensor(xdoe, **tkwargs)
ydoe = objective.evaluate(xdoe)
ydoe = ydoe.reshape((ydoe.shape[0], objective.output_dim))
n_iterations = 10

# Calculating initial scores for standard BO procedure
score_doe = objective.utility(ydoe).unsqueeze(-1)

# Definition the rombo models
autoencoder = MLPAutoEnc(high_dim=ydoe.shape[-1], hidden_dims=[128,64], zd = 10, activation = torch.nn.SiLU())
rom_args = {"autoencoder": autoencoder, "low_dim_model": KroneckerMultiTaskGP, "low_dim_likelihood": ExactMarginalLogLikelihood,
            "standard": False}
optim_args = {"q": 3, "num_restarts": 10, "raw_samples": 512}
optimizer1 = ROMBO(init_x=xdoe, init_y=ydoe, num_samples=32, bounds = bounds, MCObjective=objective, acquisition=qLogExpectedImprovement, ROM=AUTOENCROM, ROM_ARGS=rom_args)
optimizer2 = ROMBO(init_x=xdoe, init_y=ydoe, num_samples=32, bounds = bounds, MCObjective=objective, acquisition=qExpectedImprovement, ROM=AUTOENCROM, ROM_ARGS=rom_args)
optimizer3 = BO(init_x=xdoe, init_y=score_doe, num_samples=32, bounds = bounds, MCObjective=objective, acquisition=qExpectedImprovement, GP=SingleTaskGP, 
                MLL=ExactMarginalLogLikelihood)

stdbo_objectives = []
romboei_objectives = []
rombologei_objectives = []

stdbo_dvs = []
romboei_dvs = []
rombologei_dvs = []

for iteration in range(n_iterations):

    print("\n\n##### Running iteration {} out of {} #####".format(iteration+1, n_iterations))

    optimizer1.do_one_step(tag = 'ROMBO + Log EI', tkwargs=optim_args)
    optimizer2.do_one_step(tag = 'ROMBO + EI', tkwargs=optim_args)
    optimizer3.do_one_step(tag = 'BO + EI', tkwargs=optim_args)

    stdbo_objectives.append(optimizer3.best_f)
    stdbo_dvs.append(optimizer3.best_x)

    romboei_objectives.append(optimizer2.best_f)
    romboei_dvs.append(optimizer2.best_x)

    rombologei_objectives.append(optimizer1.best_f)
    rombologei_dvs.append(optimizer1.best_x)

results = {"BO": {"objectives": stdbo_objectives, "design": stdbo_dvs}, "ROMBO_EI": {"objectives": romboei_objectives, "design": romboei_dvs},
            "ROMBO_LOGEI": {"objectives": rombologei_objectives, "design": rombologei_dvs}}
savemat("env_model_results_256_q3.mat", results)






