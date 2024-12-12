"Example script for building multiple ROM optimizers, solving the environment model problem and comparing the results"

# Importing standard libraries
import torch 
import time
from smt.sampling_methods import LHS
from rombo.rom.nonlinrom import AUTOENCROM
import numpy as np
from rombo.dimensionality_reduction.autoencoder import MLPAutoEnc
from rombo.test_problems.test_problems import EnvModelFunction
from rombo.optimization.rombo import ROMBO_BAxUS
from rombo.optimization.baxus_utils import BaxusState, embedding_matrix
from rombo.optimization.stdbo import BO
from scipy.io import savemat, loadmat
import argparse
import warnings
warnings.filterwarnings('ignore')

# Importing relevant classes from BoTorch
from botorch.acquisition import qLogExpectedImprovement
from botorch.models import KroneckerMultiTaskGP, SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

tkwargs = {"device": torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0"), "dtype": torch.float64}

# Parsing input and output dim
parser = argparse.ArgumentParser()
parser.add_argument("--input_dim", help="input dimension of the function",type=int)
parser.add_argument("--output_dim", help="output dimension of the function",type=int)
args = parser.parse_args()

# Creating the initial design of experiments
inputdim = args.input_dim
xlimits = np.array([[-1.0, 1.0]]*inputdim)
n_init = 10
objective = EnvModelFunction(input_dim=inputdim, output_dim=args.output_dim, baxus_norm=True)
bounds = torch.cat((torch.zeros(1, inputdim), torch.ones(1, inputdim))).to(**tkwargs)
n_trials = 2
n_iterations = 2

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

for trial in range(n_trials):

    print("\n\n##### Running trial {} out of {} #####".format(trial+1, n_trials))

    state = BaxusState(dim=inputdim, eval_budget=n_iterations)
    S = embedding_matrix(input_dim=state.dim, target_dim=state.d_init)
    sampler = LHS(xlimits=xlimits, criterion="ese")
    xdoe = sampler(n_init)
    xdoe = torch.tensor(xdoe, **tkwargs)
    ydoe = objective.evaluate(xdoe)
    ydoe = ydoe.reshape((ydoe.shape[0], objective.output_dim))

    # Calculating initial scores for standard BO procedure
    score_doe = objective.utility(ydoe).unsqueeze(-1)

    # Definition the rombo models
    autoencoder = MLPAutoEnc(high_dim=ydoe.shape[-1], hidden_dims=[256,64], zd = 10, activation = torch.nn.SiLU())
    rom_args = {"autoencoder": autoencoder, "low_dim_model": KroneckerMultiTaskGP, "low_dim_likelihood": ExactMarginalLogLikelihood,
                "standard": False, "saas": True}
    optim_args = {"q": 1, "num_restarts": 5, "raw_samples": 512}
    optimizer1 = ROMBO_BAxUS(init_x=xdoe, init_y=ydoe, num_samples=32, bounds = bounds, MCObjective=objective, 
                             acquisition=qLogExpectedImprovement, ROM=AUTOENCROM, ROM_ARGS=rom_args, state=state, embedding=S)

    for iteration in range(n_iterations):

        print("\n\n##### Running iteration {} out of {} #####".format(iteration+1, n_iterations))

        ti = time.time()
        optimizer1.do_one_step(tag = 'ROMBO + Log EI', tkwargs=optim_args)
        tf = time.time()
        rombologei_t[trial][iteration] = tf-ti
    
        rombologei_objectives[trial][iteration] = optimizer1.best_f
        rombologei_dvs[trial][iteration] = optimizer1.best_x
    
    rombologei_doe[trial] = optimizer1.xdoe
 
results = {"BO_EI": {"objectives": boei_objectives, "design": boei_dvs, "doe": boei_doe, "time": boei_t}, "BO_LOGEI": {"objectives": bologei_objectives, "design": bologei_dvs, "doe": bologei_doe, "time": bologei_t}, 
           "ROMBO_EI": {"objectives": romboei_objectives, "design": romboei_dvs, "doe": romboei_doe, "time": romboei_t}, "ROMBO_LOGEI": {"objectives": rombologei_objectives, "design": rombologei_dvs, "doe": rombologei_doe, "time": rombologei_t}}
savemat("env_model_results_{}_{}_BaXUS.mat".format(args.input_dim, args.output_dim), results)

