import numpy as np
import torch
from aromad.dimensionality_reduction.autoencoder import MLPAutoEnc
from aromad.test_problems.test_problems import Airfoil
from aromad.optimization.altbo import ConstrainedBO
from prefoil.utils import readCoordFile
from scipy.io import savemat, loadmat
import warnings
warnings.filterwarnings('ignore')

# Importing relevant classes from BoTorch
from botorch.acquisition import qExpectedImprovement
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

# Libraries for running airfoil calculations
from blackbox import AirfoilCST
from baseclasses import AeroProblem

# Arguments for GPU-related calculations
tkwargs = {"device": torch.device("cpu"), "dtype": torch.float}

# Defining options for CFD solver, meshing and blackbox

# Flow solver options
solverOptions = {
    # Common Parameters
    "monitorvariables": ["cl", "cd", "cmz", "yplus"],
    "surfaceVariables": ["cp", "cf", "cfx", "cfy", "cfz"],
    "writeTecplotSurfaceSolution": True,
    "writeSurfaceSolution": False,
    "writeVolumeSolution": False,
    # Physics Parameters
    "equationType": "RANS",
    "smoother": "DADI",
    "MGCycle": "sg",
    "nsubiterturb": 10,
    "nCycles": 10000,
    # ANK Solver Parameters
    "useANKSolver": True,
    "ANKJacobianLag": 5,
    "ANKPhysicalLSTol": 0.25,
    "ANKOuterPreconIts": 2,
    "ANKInnerPreconIts": 2,
    "ANKASMOverlap": 2,
    "ANKSecondOrdSwitchTol": 1e-3,
    "ANKCFLLimit":1e3,
    # NK Solver Parameters
    "useNKSolver": True,
    "NKSwitchTol": 1e-8,
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
    "N": 129,
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
    #"alpha":"implicit",
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

def area_constraint(x):
    
    area = airfoil.calculateArea(x)
    return 1 - area/base_area

# Importing global x-coordinates
airfoil_file = "init_samples/1/deformedAirfoil.dat"
coords = readCoordFile(airfoil_file)
coords = np.delete(coords, -1, 0)
x_c = coords[:,0]
CL_target = 0.824
base_area = 0.07766296041532503

# Defining blackbox object
airfoil = AirfoilCST(options=options)

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

# Defining the problem
problem = Airfoil(directory="./init_samples", airfoil=airfoil, airfoil_x=x_c, upper_bounds=upperBounds, lower_bounds=lowerBounds, targetCL=CL_target, 
                  base_area=base_area, base_thickness = 0.0, baseline_upper=airfoil.DVGeo.defaultDV['upper'], baseline_lower=airfoil.DVGeo.defaultDV['lower'], 
                  tkwargs=tkwargs, normalized = False)

optimizer = ConstrainedBO(obj_x=problem.xdoe, obj_y=problem.coefdrag, cons_x = [problem.xdoe,problem.xdoe], cons_y = [problem.coeflift, problem.area],
                          n_unknown_cons=1, cons_limit=[CL_target, base_area], cons_known=[area_constraint],
                      num_samples=32, lowerBounds = lowerBounds, upperBounds = upperBounds, MCObjective=problem, acquisition=qExpectedImprovement, 
                      GP=SingleTaskGP, MLL=ExactMarginalLogLikelihood)

optimizer.do_one_step(tag='BO+EI')
