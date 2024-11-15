"Example script for building multiple ROM optimizers, solving the environment model problem and comparing the results"

# Importing standard libraries
import torch 
import time
from smt.sampling_methods import LHS
from rombo.rom.nonlinrom import AUTOENCROM
import numpy as np
from rombo.dimensionality_reduction.autoencoder import MLPAutoEnc
from rombo.test_problems.test_problems import EnvModelFunction
from rombo.optimization.rombo import ROMBO
from rombo.optimization.stdbo import BO
from scipy.io import savemat, loadmat
import warnings
warnings.filterwarnings('ignore')

# Importing relevant classes from BoTorch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.models import KroneckerMultiTaskGP, SingleTaskGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

tkwargs = {"device": torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0"), "dtype": torch.float}

# Creating the initial design of experiments
inputdim = 15
xlimits = np.array([[0.0, 1.0]]*inputdim)
n_init = 10
objective = EnvModelFunction(input_dim=inputdim, output_dim=1024, normalized=True)
bounds = torch.cat((torch.zeros(1, inputdim), torch.ones(1, inputdim))).to(**tkwargs)
n_trials = 1
n_iterations = 40

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

# boei_predictions = np.zeros((n_trials, n_iterations))
# bologei_predictions = np.zeros((n_trials, n_iterations))
# romboei_predictions = np.zeros((n_trials, n_iterations, 64))
# rombologei_predictions = np.zeros((n_trials, n_iterations, 64))

for trial in range(n_trials):

    print("\n\n##### Running trial {} out of {} #####".format(trial+1, n_trials))

    sampler = LHS(xlimits=xlimits, criterion="ese", random_state = 25)
    xdoe = sampler(n_init)
    xdoe = torch.tensor(xdoe, **tkwargs)
    ydoe = objective.evaluate(xdoe)
    ydoe = ydoe.reshape((ydoe.shape[0], objective.output_dim))

    # Calculating initial scores for standard BO procedure
    score_doe = objective.utility(ydoe).unsqueeze(-1)

    # Definition the rombo models
    autoencoder = MLPAutoEnc(high_dim=ydoe.shape[-1], hidden_dims=[256,64], zd = 10, activation = torch.nn.SiLU())
    rom_args = {"autoencoder": autoencoder, "low_dim_model": SaasFullyBayesianMultiTaskGP, "low_dim_likelihood": ExactMarginalLogLikelihood,
                "standard": False, "saas": True}
    optim_args = {"q": 1, "num_restarts": 10, "raw_samples": 512}
    optimizer1 = ROMBO(init_x=xdoe, init_y=ydoe, num_samples=32, bounds = bounds, MCObjective=objective, acquisition=qLogExpectedImprovement, ROM=AUTOENCROM, ROM_ARGS=rom_args)
    optimizer2 = ROMBO(init_x=xdoe, init_y=ydoe, num_samples=32, bounds = bounds, MCObjective=objective, acquisition=qExpectedImprovement, ROM=AUTOENCROM, ROM_ARGS=rom_args)
    optimizer3 = BO(init_x=xdoe, init_y=score_doe, num_samples=32, bounds = bounds, MCObjective=objective, acquisition=qExpectedImprovement, GP=SaasFullyBayesianSingleTaskGP, 
                    MLL=ExactMarginalLogLikelihood, training='bayesian')
    optimizer4 = BO(init_x=xdoe, init_y=score_doe, num_samples=32, bounds = bounds, MCObjective=objective, acquisition=qLogExpectedImprovement, GP=SaasFullyBayesianSingleTaskGP, 
                    MLL=ExactMarginalLogLikelihood, training='bayesian')

    for iteration in range(n_iterations):

        print("\n\n##### Running iteration {} out of {} #####".format(iteration+1, n_iterations))

        ti = time.time()
        optimizer1.do_one_step(tag = 'ROMBO + Log EI', tkwargs=optim_args)
        tf = time.time()
        rombologei_t[trial][iteration] = tf-ti
        ti = time.time()
        optimizer2.do_one_step(tag = 'ROMBO + EI', tkwargs=optim_args)
        tf = time.time()
        romboei_t[trial][iteration] = tf-ti
        ti = time.time()
        optimizer3.do_one_step(tag = 'BO + EI', tkwargs=optim_args)
        tf = time.time()
        boei_t[trial][iteration] = tf-ti
        optimizer4.do_one_step(tag = 'BO + Log EI', tkwargs=optim_args)

        boei_objectives[trial][iteration] = optimizer3.best_f
        boei_dvs[trial][iteration] = optimizer3.best_x

        bologei_objectives[trial][iteration] = optimizer4.best_f
        bologei_dvs[trial][iteration] = optimizer4.best_x

        romboei_objectives[trial][iteration] = optimizer2.best_f
        romboei_dvs[trial][iteration] = optimizer2.best_x

        rombologei_objectives[trial][iteration] = optimizer1.best_f
        rombologei_dvs[trial][iteration] = optimizer1.best_x

results = {"BO_EI": {"objectives": boei_objectives, "design": boei_dvs, "xdoe": optimizer3.xdoe, "ydoe": optimizer3.ydoe, "time": boei_t}, "BO_LOGEI": {"objectives": bologei_objectives, "design": bologei_dvs, "xdoe": optimizer4.xdoe, "ydoe": optimizer4.ydoe, "time": bologei_t}, 
           "ROMBO_EI": {"objectives": romboei_objectives, "design": romboei_dvs, "xdoe": optimizer2.xdoe, "ydoe": optimizer2.ydoe, "time": romboei_t}, "ROMBO_LOGEI": {"objectives": rombologei_objectives, "design": rombologei_dvs, "xdoe": optimizer1.xdoe, "ydoe": optimizer1.ydoe, "time": rombologei_t}}
savemat("env_model_results_1024_v1_w_Saasbo.mat", results)

