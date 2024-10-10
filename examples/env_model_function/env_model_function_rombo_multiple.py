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
n_init = 10
objective = EnvModelFunction(input_dim=15, output_dim=64, normalized=True)
#bounds = torch.tensor([[7.0, 0.02, 0.01, 30.010], [13.0, 0.12, 3.00, 30.295]], **tkwargs)
bounds = torch.cat((torch.zeros(1, 15), torch.ones(1, 15))).to(**tkwargs)
n_trials = 1
n_iterations = 30

boei_objectives = np.zeros((n_trials, n_iterations))
bologei_objectives = np.zeros((n_trials, n_iterations))
romboei_objectives = np.zeros((n_trials, n_iterations))
rombologei_objectives = np.zeros((n_trials, n_iterations))

boei_dvs = np.zeros((n_trials, n_iterations))
bologei_dvs = np.zeros((n_trials, n_iterations))
romboei_dvs = np.zeros((n_trials, n_iterations))
rombologei_dvs = np.zeros((n_trials, n_iterations))

for trial in range(n_trials):

    print("\n\n##### Running trial {} out of {} #####".format(trial+1, n_trials))

    sampler = LHS(xlimits=xlimits, criterion="ese")
    xdoe = sampler(n_init)
    xdoe = torch.tensor(xdoe, **tkwargs)
    ydoe = objective.evaluate(xdoe)
    ydoe = ydoe.reshape((ydoe.shape[0], objective.output_dim))

    # Calculating initial scores for standard BO procedure
    score_doe = objective.utility(ydoe).unsqueeze(-1)

    # Definition the rombo models
    autoencoder = MLPAutoEnc(high_dim=ydoe.shape[-1], hidden_dims=[128,64], zd = 10, activation = torch.nn.SiLU())
    rom_args = {"autoencoder": autoencoder, "low_dim_model": KroneckerMultiTaskGP, "low_dim_likelihood": ExactMarginalLogLikelihood,
                "standard": False}
    optim_args = {"q": 1, "num_restarts": 10, "raw_samples": 512}
    optimizer1 = ROMBO(init_x=xdoe, init_y=ydoe, num_samples=32, bounds = bounds, MCObjective=objective, acquisition=qLogExpectedImprovement, ROM=AUTOENCROM, ROM_ARGS=rom_args)
    optimizer2 = ROMBO(init_x=xdoe, init_y=ydoe, num_samples=32, bounds = bounds, MCObjective=objective, acquisition=qExpectedImprovement, ROM=AUTOENCROM, ROM_ARGS=rom_args)
    optimizer3 = BO(init_x=xdoe, init_y=score_doe, num_samples=32, bounds = bounds, MCObjective=objective, acquisition=qExpectedImprovement, GP=SingleTaskGP, 
                    MLL=ExactMarginalLogLikelihood)
    optimizer4 = BO(init_x=xdoe, init_y=score_doe, num_samples=32, bounds = bounds, MCObjective=objective, acquisition=qLogExpectedImprovement, GP=SingleTaskGP, 
                    MLL=ExactMarginalLogLikelihood)

    for iteration in range(n_iterations):

        print("\n\n##### Running iteration {} out of {} #####".format(iteration+1, n_iterations))

        optimizer1.do_one_step(tag = 'ROMBO + Log EI', tkwargs=optim_args)
        optimizer2.do_one_step(tag = 'ROMBO + EI', tkwargs=optim_args)
        optimizer3.do_one_step(tag = 'BO + EI', tkwargs=optim_args)
        optimizer4.do_one_step(tag = 'BO + Log EI', tkwargs=optim_args)

        boei_objectives[trial][iteration] = optimizer3.best_f
        boei_dvs[trial][iteration] = optimizer3.best_x

        bologei_objectives[trial][iteration] = optimizer4.best_f
        bologei_dvs[trial][iteration] = optimizer4.best_x

        romboei_objectives[trial][iteration] = optimizer2.best_f
        romboei_dvs[trial][iteration] = optimizer2.best_x

        rombologei_objectives[trial][iteration] = optimizer1.best_f
        rombologei_dvs[trial][iteration] = optimizer1.best_x

results = {"BO_EI": {"objectives": boei_objectives, "design": boei_dvs, "xdoe": optimizer3.xdoe, "ydoe": optimizer3.ydoe}, "BO_LOGEI": {"objectives": bologei_objectives, "design": bologei_dvs, "xdoe": optimizer4.xdoe, "ydoe": optimizer4.ydoe}, 
           "ROMBO_EI": {"objectives": romboei_objectives, "design": romboei_dvs, "xdoe": optimizer2.xdoe, "ydoe": optimizer2.ydoe}, "ROMBO_LOGEI": {"objectives": rombologei_objectives, "design": rombologei_dvs, "xdoe": optimizer1.xdoe, "ydoe": optimizer1.ydoe}}
savemat("env_model_results_64_v1.mat", results)

