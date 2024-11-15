"Example script for building multiple ROM optimizers, solving the bursselator PDE problem and comparing the results"

# Importing standard libraries
import torch 
import time
from smt.sampling_methods import LHS
from rombo.rom.nonlinrom import AUTOENCROM
import numpy as np
from rombo.dimensionality_reduction.autoencoder import MLPAutoEnc
from rombo.test_problems.test_problems import BrusselatorPDE
from rombo.optimization.rombo import ROMBO
from rombo.optimization.stdbo import BO
from scipy.io import savemat, loadmat
import warnings
warnings.filterwarnings('ignore')

# Importing relevant classes from BoTorch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.models import KroneckerMultiTaskGP, SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

tkwargs = {"device": torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0"), "dtype": torch.float}

# Defining the objective class and bounds
objective = BrusselatorPDE(input_dim=32, Nx=64, Ny=64, tkwargs=tkwargs)
bounds = torch.cat((torch.zeros(1, 32), torch.ones(1, 32))).to(**tkwargs)

# Generating the design of experiments
n_init = 5
xlimits = np.array([[0.0,1.0]]*32)

# Defining the optimization parameters
n_iterations = 2
n_trials = 2

boei_objectives = np.zeros((n_trials, n_iterations))
bologei_objectives = np.zeros((n_trials, n_iterations))
romboei_objectives = np.zeros((n_trials, n_iterations))
rombologei_objectives = np.zeros((n_trials, n_iterations))

boei_dvs = np.zeros((n_trials, n_iterations))
bologei_dvs = np.zeros((n_trials, n_iterations))
romboei_dvs = np.zeros((n_trials, n_iterations))
rombologei_dvs = np.zeros((n_trials, n_iterations))

boei_time = np.zeros((n_trials, n_iterations))
bologei_time = np.zeros((n_trials, n_iterations))
romboei_time = np.zeros((n_trials, n_iterations))
rombologei_time = np.zeros((n_trials, n_iterations))

for trial in range(n_trials):

    print("\n\n##### Running trial {} out of {} #####".format(trial+1, n_trials))

    sampler = LHS(xlimits=xlimits, criterion="ese")
    xdoe = sampler(n_init)
    ydoe = objective.evaluate(xdoe)
    ydoe = ydoe.reshape((ydoe.shape[0], objective.output_dim))
    xdoe = torch.tensor(xdoe, **tkwargs)
    ydoe = torch.tensor(ydoe, **tkwargs)

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

        t1 = time.time()
        optimizer1.do_one_step(tag = 'ROMBO + Log EI', tkwargs=optim_args)
        t2 = time.time()
        romboei_time[trial][iteration] = t2 - t1

        t1 = time.time()
        optimizer2.do_one_step(tag = 'ROMBO + EI', tkwargs=optim_args)
        t2 = time.time()
        rombologei_time[trial][iteration] = t2 - t1
        
        t1 = time.time()
        optimizer3.do_one_step(tag = 'BO + EI', tkwargs=optim_args)
        t2 = time.time()
        boei_time[trial][iteration] = t2 - t1

        t1 = time.time()
        optimizer4.do_one_step(tag = 'BO + Log EI', tkwargs=optim_args)
        t2 = time.time()
        bologei_time[trial][iteration] = t2 - t1

        boei_objectives[trial][iteration] = optimizer3.best_f
        boei_dvs[trial][iteration] = optimizer3.best_x

        bologei_objectives[trial][iteration] = optimizer4.best_f
        bologei_dvs[trial][iteration] = optimizer4.best_x

        romboei_objectives[trial][iteration] = optimizer2.best_f
        romboei_dvs[trial][iteration] = optimizer2.best_x

        rombologei_objectives[trial][iteration] = optimizer1.best_f
        rombologei_dvs[trial][iteration] = optimizer1.best_x

results = {"BO_EI": {"objectives": boei_objectives, "design": boei_dvs, "time": boei_time, "xdoe": optimizer3.xdoe.detach().cpu().numpy(), "ydoe": optimizer3.ydoe.detach().cpu().numpy()}, 
        "BO_LOGEI": {"objectives": bologei_objectives, "design": bologei_dvs, "time": bologei_time, "xdoe": optimizer4.xdoe.detach().cpu().numpy(), "ydoe": optimizer4.ydoe.detach().cpu().numpy()}, 
           "ROMBO_EI": {"objectives": romboei_objectives, "design": romboei_dvs, "time": romboei_time, "xdoe": optimizer2.xdoe.detach().cpu().numpy(), "ydoe": optimizer2.ydoe.detach().cpu().numpy()}, 
           "ROMBO_LOGEI": {"objectives": rombologei_objectives, "design": rombologei_dvs, "time":rombologei_time, "xdoe": optimizer1.xdoe.detach().cpu().numpy(), "ydoe": optimizer1.ydoe.detach().cpu().numpy()}}
savemat("/scratch/gilbreth/adikshit/brusselatorPDE_64_multiple_trials.mat", results)






