# Importing standard libraries
import torch 
import time
from smt.sampling_methods import LHS
from rombo.rom.linrom import PODROM
import numpy as np
from rombo.test_problems.test_problems import EnvModelFunction
from rombo.optimization.rombo import ROMBO
from rombo.optimization.stdbo import BO
from botorch.models.transforms import Standardize
from scipy.io import savemat, loadmat
import argparse
import warnings
warnings.filterwarnings('ignore')

# Importing relevant classes from BoTorch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.models import SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

tkwargs = {"device": torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0"), "dtype": torch.float64}

# Parsing parameters from the command line
parser = argparse.ArgumentParser()
parser.add_argument("--input_dim", help="input dimension of the function", type=int)
parser.add_argument("--output_dim", help="output dimension of the function", type=int)
parser.add_argument("--mc_samples", help="number of MC samples", type=int)
parser.add_argument("--trial_num", help="number of the trial of the program",type=int)
args = parser.parse_args()

# Instantiating the problem and defining optimization parameters
inputdim = args.input_dim
xlimits = np.array([[0.0, 1.0]]*inputdim)
n_init = 10
objective = EnvModelFunction(input_dim=inputdim, output_dim=args.output_dim, normalized=True)
bounds = torch.cat((torch.zeros(1, inputdim), torch.ones(1, inputdim))).to(**tkwargs)
n_trials = 1
n_iterations = 40

# Defining arrays to store values during the optimization loop
romboei_objectives = np.zeros((n_trials, n_iterations))
rombologei_objectives = np.zeros((n_trials, n_iterations))

romboei_dvs = np.zeros((n_trials, n_iterations))
rombologei_dvs = np.zeros((n_trials, n_iterations))

romboei_t = np.zeros((n_trials, n_iterations))
rombologei_t = np.zeros((n_trials, n_iterations))

romboei_doe = np.zeros((n_trials, n_iterations+n_init, inputdim))
rombologei_doe = np.zeros((n_trials, n_iterations+n_init, inputdim))

romboei_EI = np.zeros((n_trials, n_iterations))
rombologei_EI = np.zeros((n_trials, n_iterations))

romboei_k = np.zeros((n_trials, n_iterations))
rombologei_k = np.zeros((n_trials, n_iterations))

for trial in range(n_trials):

    print("\n\n##### Running trial {} out of {} #####".format(trial+1, n_trials))
    
    # Generating the initial samples for the trial
    sampler = LHS(xlimits=xlimits, criterion="ese", random_state=args.trial_num)
    xdoe = sampler(n_init)
    xdoe = torch.tensor(xdoe, **tkwargs)
    ydoe = objective.evaluate(xdoe)
    ydoe = ydoe.reshape((ydoe.shape[0], objective.output_dim))

    # Calculating initial scores for standard BO procedure
    score_doe = objective.utility(ydoe).unsqueeze(-1)

    # Definition the podbo optimizers as a special case of the rombo optimization class
    rom_args = {"ric": 0.9999, "low_dim_model": SingleTaskGP, "low_dim_likelihood": SumMarginalLogLikelihood,
                "saas": False}
    optim_args = {"q": 1, "num_restarts": 25, "raw_samples": 512}
    optimizer1 = ROMBO(init_x=xdoe, init_y=ydoe, num_samples=args.mc_samples, bounds = bounds, MCObjective=objective, acquisition=qLogExpectedImprovement, ROM=PODROM, ROM_ARGS=rom_args)
    optimizer2 = ROMBO(init_x=xdoe, init_y=ydoe, num_samples=args.mc_samples, bounds = bounds, MCObjective=objective, acquisition=qExpectedImprovement, ROM=PODROM, ROM_ARGS=rom_args)

    # Running the optimization loop
    for iteration in range(n_iterations):

        print("\n\n##### Running iteration {} out of {} #####".format(iteration+1, n_iterations))

        ti = time.time()
        optimizer1.do_one_step(tag = 'PODBO + Log EI', tkwargs=optim_args)
        tf = time.time()
        rombologei_t[trial][iteration] = tf-ti
        ti = time.time()
        optimizer2.do_one_step(tag = 'PODBO + EI', tkwargs=optim_args)
        tf = time.time()
        romboei_t[trial][iteration] = tf-ti

        romboei_objectives[trial][iteration] = optimizer2.best_f
        romboei_dvs[trial][iteration] = optimizer2.best_x
        romboei_EI[trial][iteration] = optimizer2.maxEI
        romboei_k[trial][iteration] = optimizer2.rom_model.k

        rombologei_objectives[trial][iteration] = optimizer1.best_f
        rombologei_dvs[trial][iteration] = optimizer1.best_x
        rombologei_EI[trial][iteration] = optimizer1.maxEI
        rombologei_k[trial][iteration] = optimizer1.rom_model.k
    
    romboei_doe[trial] = optimizer2.xdoe.detach().cpu().numpy()
    rombologei_doe[trial] = optimizer1.xdoe.detach().cpu().numpy()

# Saving the final data
results = {"ROMBO_EI": {"objectives": romboei_objectives, "design": romboei_dvs, "doe": romboei_doe, "n_modes": romboei_k, "time": romboei_t}, 
            "ROMBO_LOGEI": {"objectives": rombologei_objectives, "design": rombologei_dvs, "doe": rombologei_doe, "n_modes": rombologei_k, "time": rombologei_t}}
savemat("./env_model_results_{}_{}_POD_BO_trial_{}_MC_{}.mat".format(args.input_dim, args.output_dim, args.trial_num, args.mc_samples), results)