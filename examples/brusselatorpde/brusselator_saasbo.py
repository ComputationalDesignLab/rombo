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
import argparse
from scipy.io import savemat, loadmat
import warnings
warnings.filterwarnings('ignore')

# Importing relevant classes from BoTorch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

tkwargs = {"device": torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0"), "dtype": torch.float64}

# Parsing parameters from the command line
parser = argparse.ArgumentParser()
parser.add_argument("--input_dim", help="input dimension of the function",type=int)
parser.add_argument("--optimizer", help="optimizer to be run - 1:ROMBO+LogEI, 2:ROMBO+EI, 3:BO",type=int)
parser.add_argument("--trial_num", help="number of the trial run",type=int)
args = parser.parse_args()

# Instantiating the problem and setting the optimization parameters
inputdim = args.input_dim
xlimits = np.array([[0.0, 1.0]]*inputdim)
n_init = 5
objective = BrusselatorPDE(input_dim=inputdim, Nx=64, Ny=64)
bounds = torch.cat((torch.zeros(1, inputdim), torch.ones(1, inputdim))).to(**tkwargs)
n_trials = 1
n_iterations = 25

# Defining arrays to store values during the optimization loop
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

boei_doe = np.zeros((n_trials, n_iterations+n_init, inputdim))
bologei_doe = np.zeros((n_trials, n_iterations+n_init, inputdim))
romboei_doe = np.zeros((n_trials, n_iterations+n_init, inputdim))
rombologei_doe = np.zeros((n_trials, n_iterations+n_init, inputdim))

boei_predictions = np.zeros((n_trials, n_iterations))
bologei_predictions = np.zeros((n_trials, n_iterations))
romboei_predictions = np.zeros((n_trials, n_iterations, 8192))
rombologei_predictions = np.zeros((n_trials, n_iterations, 8192))

boei_EI = np.zeros((n_trials, n_iterations))
bologei_EI = np.zeros((n_trials, n_iterations))
romboei_EI = np.zeros((n_trials, n_iterations))
rombologei_EI = np.zeros((n_trials, n_iterations))

boei_lengthscales = np.zeros((n_trials, n_iterations, inputdim))
bologei_lengthscales = np.zeros((n_trials, n_iterations, inputdim))
romboei_lengthscales = np.zeros((n_trials, n_iterations, inputdim))
rombologei_lengthscales = np.zeros((n_trials, n_iterations, inputdim))

for trial in range(n_trials):

    print("\n\n##### Running trial {} out of {} #####".format(trial+1, n_trials))

    # Generating the initial design of experiments
    sampler = LHS(xlimits=xlimits, criterion="ese", random_state = args.trial_num)
    xdoe = sampler(n_init)
    xdoe = torch.tensor(xdoe, **tkwargs)
    ydoe = objective.evaluate(xdoe)
    ydoe = ydoe.reshape((ydoe.shape[0], objective.output_dim))

    # Calculating initial scores for standard BO procedure
    score_doe = objective.utility(ydoe).unsqueeze(-1)

    # Defining the BO optimizers
    autoencoder = MLPAutoEnc(high_dim=ydoe.shape[-1], hidden_dims=[512,128,64], zd = 10, activation = torch.nn.SiLU())
    autoencoder.double()
    rom_args = {"autoencoder": autoencoder, "low_dim_model": SaasFullyBayesianMultiTaskGP, "low_dim_likelihood": ExactMarginalLogLikelihood,
                "standard": False, "saas": True}
    optim_args = {"q": 1, "num_restarts": 25, "raw_samples": 512}
    optimizer1 = ROMBO(init_x=xdoe, init_y=ydoe, num_samples=32, bounds = bounds, MCObjective=objective, acquisition=qLogExpectedImprovement, ROM=AUTOENCROM, ROM_ARGS=rom_args)
    optimizer2 = ROMBO(init_x=xdoe, init_y=ydoe, num_samples=32, bounds = bounds, MCObjective=objective, acquisition=qExpectedImprovement, ROM=AUTOENCROM, ROM_ARGS=rom_args)
    optimizer3 = BO(init_x=xdoe, init_y=score_doe, num_samples=32, bounds = bounds, MCObjective=objective, acquisition=qExpectedImprovement, GP=SaasFullyBayesianSingleTaskGP, 
                    MLL=ExactMarginalLogLikelihood, training='bayesian')
    optimizer4 = BO(init_x=xdoe, init_y=score_doe, num_samples=32, bounds = bounds, MCObjective=objective, acquisition=qLogExpectedImprovement, GP=SaasFullyBayesianSingleTaskGP, 
                    MLL=ExactMarginalLogLikelihood, training='bayesian')

    # Running the optimization loop
    for iteration in range(n_iterations):

        print("\n\n##### Running iteration {} out of {} #####".format(iteration+1, n_iterations))

        if args.optimizer == 1:
            ti = time.time()
            optimizer1.do_one_step(tag = 'ROMBO + Log EI', tkwargs=optim_args)
            tf = time.time()
            rombologei_t[trial][iteration] = tf-ti
            rombologei_objectives[trial][iteration] = optimizer1.best_f
            rombologei_dvs[trial][iteration] = optimizer1.best_x
            rombologei_EI[trial][iteration] = optimizer1.maxEI

        if args.optimizer == 2:
            ti = time.time()
            optimizer2.do_one_step(tag = 'ROMBO + EI', tkwargs=optim_args)
            tf = time.time()
            romboei_t[trial][iteration] = tf-ti
            romboei_objectives[trial][iteration] = optimizer2.best_f
            romboei_dvs[trial][iteration] = optimizer2.best_x
            romboei_EI[trial][iteration] = optimizer2.maxEI

        if args.optimizer == 3:
            ti = time.time()
            optimizer3.do_one_step(tag = 'BO + EI', tkwargs=optim_args)
            tf = time.time()
            boei_objectives[trial][iteration] = optimizer3.best_f
            boei_dvs[trial][iteration] = optimizer3.best_x
            boei_EI[trial][iteration] = optimizer3.maxEI
            boei_t[trial][iteration] = tf-ti

            ti = time.time()
            optimizer4.do_one_step(tag = 'BO + Log EI', tkwargs=optim_args)
            tf = time.time()
            bologei_objectives[trial][iteration] = optimizer4.best_f
            bologei_dvs[trial][iteration] = optimizer4.best_x
            bologei_EI[trial][iteration] = optimizer4.maxEI
            bologei_t[trial][iteration] = tf-ti

    if args.optimizer == 3:
        boei_doe[trial] = optimizer3.xdoe.detach().cpu().numpy()
        bologei_doe[trial] = optimizer4.xdoe.detach().cpu().numpy()
    elif args.optimizer == 2:
        romboei_doe[trial] = optimizer2.xdoe.detach().cpu().numpy()
    elif args.optimizer == 1:
        rombologei_doe[trial] = optimizer1.xdoe.detach().cpu().numpy()

# Saving the final data
results = {"BO_EI": {"objectives": boei_objectives, "design": boei_dvs, "doe": boei_doe, "time": boei_t, "predictions": boei_predictions, "EI": boei_EI}, 
            "BO_LOGEI": {"objectives": bologei_objectives, "design": bologei_dvs, "doe": bologei_doe, "time": bologei_t, "predictions": bologei_predictions, "EI": bologei_EI}, 
           "ROMBO_EI": {"objectives": romboei_objectives, "design": romboei_dvs, "doe": romboei_doe, "time": romboei_t, "predictions": romboei_predictions, "EI": romboei_EI}, 
           "ROMBO_LOGEI": {"objectives": rombologei_objectives, "design": rombologei_dvs, "doe": rombologei_doe, "time": rombologei_t, "predictions": rombologei_predictions, "EI": rombologei_EI}}
savemat("./brusselator_results_{}_optimizer{}_saasbo_trial_{}.mat".format(args.input_dim, args.optimizer, args.trial_num), results)
