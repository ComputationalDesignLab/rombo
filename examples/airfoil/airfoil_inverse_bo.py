import numpy as np
import torch
from aromad.dimensionality_reduction.autoencoder import MLPAutoEnc
from aromad.test_problems.test_problems import InverseAirfoil
from aromad.optimization.stdbo import BO
from aromad.rom.nonlinrom import AUTOENCROM
from aromad.dimensionality_reduction.autoencoder import MLPAutoEnc
from aromad.optimization.rombo import ROMBO
from scipy.io import savemat, loadmat
import warnings
warnings.filterwarnings('ignore')

# Importing relevant classes from BoTorch
from botorch.acquisition import qExpectedImprovement
from botorch.models import SingleTaskGP, KroneckerMultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

# Libraries for running airfoil calculations
from blackbox import AirfoilCST
from baseclasses import AeroProblem

# Arguments for GPU-related calculations
tkwargs = {"device": torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0"), "dtype": torch.float}

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
    #"ANKUseTurbDADI": False,
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
    name="ap", alpha=2.0, mach=0.734, reynolds=6.5e6, reynoldsLength=1.0, T=255.556,
    areaRef=1.0, chordRef=1.0, evalFuncs=["cl", "cd", "cdp", "cdv","cmz"], xRef = 0.25, yRef = 0.0, zRef = 0.0
)

# Options for blackbox
options = {
    "solverOptions": solverOptions,
    "directory": "infill_samples",
    "noOfProcessors": 8,
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

keys = ["upper", "lower"]
airfoil.addDV("alpha", lowerBound=1.5, upperBound=3.5)
# Defining bounds of the sampling problem
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

bounds = torch.tensor([lowerBounds, upperBounds], **tkwargs)

# Defining the problem
problem = InverseAirfoil(directory="./50_samples_rae2822", airfoil=airfoil, targetCp=gbo_cp, upper_bounds=upperBounds, lower_bounds=lowerBounds)

autoencoder = MLPAutoEnc(high_dim=problem.coefpressure.shape[-1], hidden_dims=[256,64], zd = 10, activation = torch.nn.SiLU())
rom_args = {"autoencoder": autoencoder, "low_dim_model": KroneckerMultiTaskGP, "low_dim_likelihood": ExactMarginalLogLikelihood,
                "standard": False}

optimizer1 = BO(init_x=problem.xdoe, init_y=problem.ydoe.unsqueeze(-1), num_samples=32, bounds=bounds, MCObjective=problem, acquisition=qExpectedImprovement, 
                      GP=SingleTaskGP, MLL=ExactMarginalLogLikelihood)
optimizer2 = ROMBO(init_x=problem.xdoe, init_y=problem.coefpressure, num_samples=32, bounds = bounds, MCObjective=problem, acquisition=qExpectedImprovement, ROM=AUTOENCROM, ROM_ARGS=rom_args)

optim_args = {"q": 1, "num_restarts": 10, "raw_samples": 512}

optimizer2.do_one_step(tag='ROMBO+EI', tkwargs=optim_args)
optimizer1.do_one_step(tag='BO+EI', tkwargs=optim_args)


