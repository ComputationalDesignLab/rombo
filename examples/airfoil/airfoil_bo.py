# Importing standard libraries
import numpy as np
import torch
import time
from rombo.dimensionality_reduction.autoencoder import MLPAutoEnc
from rombo.test_problems.test_problems import InverseAirfoil
from rombo.optimization.stdbo import BO
from botorch.models.transforms import Standardize
from rombo.rom.nonlinrom import AUTOENCROM
from rombo.dimensionality_reduction.autoencoder import MLPAutoEnc
from rombo.optimization.rombo import ROMBO
from scipy.io import savemat, loadmat
import argparse
import warnings
warnings.filterwarnings('ignore')

# Importing relevant classes from BoTorch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.models import SingleTaskGP, KroneckerMultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

# Libraries for running airfoil calculations
from blackbox import AirfoilCST
from baseclasses import AeroProblem

# Arguments for GPU-related calculations
tkwargs = {"device": torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0"), "dtype": torch.float64}
torch.cuda.manual_seed(20)
torch.use_deterministic_algorithms(True)
np.random.seed(20)

# Parsing parameters from the command line
parser = argparse.ArgumentParser()
parser.add_argument('--optimizer', help='which optimizer to run - 1: BO+EI, 2:BO+LogEI, 3:ROMBO+EI, 4:ROMBO+LogEI', type=int)
parser.add_argument('--trial', help='which trial number is being run', type=int)
args = parser.parse_args()

# Defining options for CFD solver, meshing and blackbox

# Flow solver options
solverOptions = {
    # Common Parameters
    "monitorvariables": ["cl", "cd", "cmz", "yplus","cdp"],
    "surfaceVariables": ["cp","cf","cfx","cfy","cfz","vx","vy","vz","rho","mach"],
    "writeTecplotSurfaceSolution": True,
    "writeSurfaceSolution": True,
    "writeVolumeSolution": True,
    # Physics Parameters
    "equationType": "RANS",
    "smoother": "DADI",
    "MGCycle": "sg",
    "nsubiterturb": 10,
    "nCycles": 10000,
    # ANK Solver Parameters
    "useANKSolver": True,
    "ANKNSubiterTurb":100,
    'ANKTurbKSPDebug': True,
    "ANKJacobianLag": 5,
    "ANKUnsteadyLSTol": 1.2,
    "ANKPhysicalLSTol": 0.30,
    "ANKOuterPreconIts": 2,
    "ANKInnerPreconIts": 2,
    "ANKASMOverlap": 2,
    "ANKSecondOrdSwitchTol": 1e-3,
    "ANKCFLLimit":1e3,
    # NK Solver Parameters
    "useNKSolver": True,
    "NKSwitchTol": 1e-6,
    "NKSubspaceSize": 400,
    "NKASMOverlap": 3,
    "NKPCILUFill": 4,
    "NKJacobianLag": 5,
    "NKOuterPreconIts": 3,
    "NKInnerPreconIts": 3,
    # Termination Criteria
    "L2Convergence": 1e-14,
    "L2ConvergenceCoarse": 1e-4
}

# Volume meshing options
meshingOptions = {
    # Input Parameters
    "unattachedEdgesAreSymmetry": False,
    "outerFaceBC": "farfield",
    "autoConnect": True,
    "BC": {1: {"jLow": "zSymm", "jHigh": "zSymm"}},
    "families": "wall",
    # Grid Parameters
    "N": 257,
    "s0": 1e-6,
    "marchDist": 100.0,
    # Pseudo Grid Parameters
    "ps0": -1.0,
    "pGridRatio": -1.0,
    "cMax": 3.0,
    # Smoothing parameters
    "epsE": 1.0,
    "epsI": 2.0,
    "theta": 3.0,
    "volCoef": 0.25,
    "volBlend": 0.0001,
    "volSmoothIter": 100,
}

# Creating aero problem
ap = AeroProblem(
    name="ap", alpha=2.0, mach=0.734, reynolds=6.5e6, reynoldsLength=1.0, T=288.15,
    areaRef=1.0, chordRef=1.0, evalFuncs=["cl", "cd", "cdp", "cdv","cmz"], xRef = 0.25, yRef = 0.0, zRef = 0.0
)

# Options for blackbox
options = {
    "solverOptions": solverOptions,
    "directory": "./infill_samples_opt{}_trial{}".format(args.optimizer, args.trial),
    "noOfProcessors": 64,
    "aeroProblem": ap,
    "airfoilFile": "rae2822_L1.dat",
    "numCST": [6, 6],
    "meshingOptions": meshingOptions,
    "refine": 0,
    "getFlowFieldData": True,
    "region": "surface",
    "writeAirfoilCoordinates": True,
    "plotAirfoil": True
}

# Defining blackbox object
airfoil = AirfoilCST(options=options)

# Target pressure distribution
gbo_data = loadmat('./GBO_results.mat')
gbo_cp = torch.tensor(gbo_data['Pressure_Dist'], **tkwargs)

# Defining bounds of the problem and adding design variables to the blackbox wrapper class
keys = ["upper", "lower"]
airfoil.addDV("alpha", lowerBound=1.5, upperBound=3.5)
lowerBounds = np.array([1.5])
upperBounds = np.array([3.5])
keys = ["upper", "lower"]
for key in keys:
    coeff = airfoil.DVGeo.defaultDV[key] # get the fitted CST coeff
    if key == "upper":
        dv_min = coeff - 0.30*coeff
        dv_max = coeff + 0.30*coeff
    else:
        dv_min = coeff - 0.30*coeff
        dv_max = coeff + 0.30*coeff
    airfoil.addDV(key, lowerBound=dv_min, upperBound=dv_max)
    for i in range(len(coeff)):
        lb = min([dv_min[i],dv_max[i]])
        ub = max([dv_min[i],dv_max[i]])
        lowerBounds = np.append(lowerBounds, lb)
        upperBounds = np.append(upperBounds, ub)

# Defining optimization bounds
bounds = torch.cat((torch.zeros(1, 13), torch.ones(1, 13))).to(**tkwargs)

# Defining the problem and optimization parameters
problem = InverseAirfoil(directory="./50_samples_rae2822_HF_{}".format(args.trial), airfoil=airfoil, targetCp=gbo_cp, 
                        upper_bounds=upperBounds, lower_bounds=lowerBounds, normalized=True)
n_trials = 1
n_iterations = 75

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

for trial in range(n_trials):

    # Defining the BO optimizers
    autoencoder = MLPAutoEnc(high_dim=problem.coefpressure.shape[-1], hidden_dims=[128,64], zd = 10, activation = torch.nn.SiLU())
    autoencoder.double()
    rom_args = {"autoencoder": autoencoder, "low_dim_model": KroneckerMultiTaskGP, "low_dim_likelihood": ExactMarginalLogLikelihood,
                    "standard": False, "saas": False}
    gp_args = {"outcome_transform": Standardize(problem.ydoe.unsqueeze(-1).shape[-1])}

    optimizer1 = BO(init_x=problem.xdoe, init_y=problem.ydoe.unsqueeze(-1), num_samples=32, bounds=bounds, MCObjective=problem, acquisition=qExpectedImprovement, 
                        GP=SingleTaskGP, GP_ARGS=gp_args, MLL=ExactMarginalLogLikelihood)
    optimizer2 = BO(init_x=problem.xdoe, init_y=problem.ydoe.unsqueeze(-1), num_samples=32, bounds=bounds, MCObjective=problem, acquisition=qLogExpectedImprovement, 
                        GP=SingleTaskGP, GP_ARGS=gp_args, MLL=ExactMarginalLogLikelihood)
    optimizer3 = ROMBO(init_x=problem.xdoe, init_y=problem.coefpressure, num_samples=32, bounds = bounds, MCObjective=problem, acquisition=qExpectedImprovement, 
                       ROM=AUTOENCROM, ROM_ARGS=rom_args)
    optimizer4 = ROMBO(init_x=problem.xdoe, init_y=problem.coefpressure, num_samples=32, bounds = bounds, MCObjective=problem, acquisition=qLogExpectedImprovement, 
                        ROM=AUTOENCROM, ROM_ARGS=rom_args)
    optim_args = {"q": 1, "num_restarts": 25, "raw_samples": 512}

    # Running the optimization loop
    for iteration in range(n_iterations):

        print("\n\n##### Running iteration {} out of {} #####".format(iteration+1, n_iterations))

        if args.optimizer == 1:
            ti = time.time()
            optimizer1.do_one_step(tag = 'BO + EI', tkwargs=optim_args)
            tf = time.time()
            boei_t[trial][iteration] = tf-ti
            boei_objectives[trial][iteration] = optimizer1.best_f
            boei_dvs[trial][iteration] = optimizer1.best_x

        elif args.optimizer == 2:
            ti = time.time()
            optimizer2.do_one_step(tag = 'BO + Log EI', tkwargs=optim_args)
            tf = time.time()
            bologei_t[trial][iteration] = tf-ti
            bologei_objectives[trial][iteration] = optimizer2.best_f
            bologei_dvs[trial][iteration] = optimizer2.best_x

        elif args.optimizer == 3:
            ti = time.time()
            optimizer3.do_one_step(tag = 'ROMBO + EI', tkwargs=optim_args)
            tf = time.time()
            romboei_t[trial][iteration] = tf-ti
            romboei_objectives[trial][iteration] = optimizer3.best_f
            romboei_dvs[trial][iteration] = optimizer3.best_x

        elif args.optimizer == 4:
            ti = time.time()
            optimizer4.do_one_step(tag = 'ROMBO + Log EI', tkwargs=optim_args)
            tf = time.time()
            rombologei_t[trial][iteration] = tf-ti
            rombologei_objectives[trial][iteration] = optimizer4.best_f
            rombologei_dvs[trial][iteration] = optimizer4.best_x

# Saving the final data
results = {"BO_EI": {"objectives": boei_objectives, "design": boei_dvs, "xdoe": optimizer1.xdoe, "ydoe": optimizer1.ydoe, "time": boei_t}, "BO_LOGEI": {"objectives": bologei_objectives, "design": bologei_dvs, "xdoe": optimizer2.xdoe, "ydoe": optimizer2.ydoe, "time": bologei_t}, 
           "ROMBO_EI": {"objectives": romboei_objectives, "design": romboei_dvs, "xdoe": optimizer3.xdoe, "ydoe": optimizer3.ydoe, "time": romboei_t}, "ROMBO_LOGEI": {"objectives": rombologei_objectives, "design": rombologei_dvs, "xdoe": optimizer4.xdoe, "ydoe": optimizer4.ydoe, "time": rombologei_t}}
savemat("./airfoil_inverse_design_optimizer{}_{}.mat".format(args.optimizer, args.trial), results)
