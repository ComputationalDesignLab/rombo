# ROMBO: A composite Bayesian optimization framework for engineering design using nonintrusive reduced-order models

<p align="center">
<img src="images/rombo-1.png"/>
</p>

ROMBO is a optimization framework that utilizes a composite function formulation and nonlinear nonintrusive reduced order models. An autoencoder neural network is used to 
project high dimensional outputs into a latent space. The latent space is modeled using multi-task Gaussian process (GP) models that utilize a Kronecker structure or intrinsic model coregionalization (ICM) formulation. The framework uses a Monte Carlo expected improvement infill strategy to balance exploration of the design space with exploitation of the objective function. A linear POD method is also implemented using the same structure as ROMBO but using POD for dimensionality reduction and independent GP models for the latent space. A standard BO implementation is also provided for generating comparison data for the ROMBO framework. The framework is built utilizing [PyTorch](https://pytorch.org/) and associated libraries such as [GPyTorch](https://gpytorch.ai/) and [BoTorch](https://botorch.org/). Modular base classes have been provided for users to implement their own ROM architectures and utilize them within this framework.

## Installation

The ROMBO code can be installed in your Python environment using pip according to the following steps:

- Clone or download the latest code from this repository. 
- Open the terminal and ``cd`` into the root of cloned/downloaded repository.
- Activate the virtual environment and run ``pip install .``
- Alternatively, run ``pip install -e .`` to install the package in development mode.

## Training a simple nonintrusive reduced order model using autoencoders and GP models

The following example code demonstrates how the ROMBO framework modules can be used to define a deep learning ROM model and train it to predict the environment model function using
the corresponding test problem class. First, the relevant modules must be imported from ROMBO and other Python packages. 

```python
import torch 
import numpy as np
from smt.sampling_methods import LHS
from rombo.dimensionality_reduction.autoencoder import MLPAutoEnc
from rombo.rom.nonlinrom import AUTOENCROM
from rombo.test_problems.test_problems import EnvModelFunction
from botorch.models import KroneckerMultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
```

A problem class is evaluated using one of the test problems defined in ROMBO. The [SMT](https://github.com/SMTorg/smt) package is used to generate a latin hypercube sampling (LHS) plan 
and the samples are evaluated using the evaluate method of the test problem class. A set of testing data is also generated in a similar manner. 

```python
# Defining environment model function 
problem = EnvModelFunction(input_dim = 15, output_dim = 1024, normalized = True)

# Creating the training data
n_data = 50
xlimits = np.array([[0.0, 1.0]]*problem.inputdim)
sampler = LHS(xlimits=xlimits, criterion="ese")
xtrain = sampler(n_data)
xtrain = torch.tensor(xtrain, **tkwargs)
htrain = problem.evaluate(xtrain).flatten(1)

# Generating the test data
test_sampler = LHS(xlimits=xlimits, criterion="ese")
xtest = test_sampler(10)
xtest = torch.tensor(xtest, **tkwargs)
htest = problem.evaluate(xtest).flatten(1)
```

The autoencoder architecture is defined using `MLPAutoEnc` which is a simple fully-connected autoencoder network defined in ROMBO. A user may also define their own architecture using [PyTorch](https://pytorch.org/) and use it along with the ROM model class within the ROMBO framework. After defining the autoencoder, the `AUTOENCROM` class can be used to define a ROM model with the corresponding training data and GP model. The GP model, `KroneckerMultiTaskGP`, and the corresponding likelihood function `ExactMarginalLogLikelihood`, are imported from the [BoTorch](https://botorch.org/) package which contains GP models built using [GPyTorch](https://gpytorch.ai/). The `AUTOENCROM` module combines the various inputs and automates the process of setting up the rom model.  

```python
# Generating the nonlinear ROM model
autoencoder = MLPAutoEnc(high_dim=problem.output_dim, hidden_dims=[256,64], zd = 10, activation = torch.nn.SiLU())
rom = AUTOENCROM(xtrain, htrain, autoencoder = autoencoder, low_dim_model = KroneckerMultiTaskGP, low_dim_likelihood = ExactMarginalLogLikelihood)
```

The ROM that is generated can be trained on the training data using the `trainROM` method and the predictions can be generated on the testing data using the `predictROM` method. 

```python
# Training the ROM and predicting on the test data
rom.trainROM(verbose=False)
field = rom.predictROM(xtest)
```

## Creating and running an optimization loop using ROMBO

To create an optimization loop using the ROMBO framework, start by importing the necessary modules and libraries into the script.

```python
# Importing standard libraries
import torch 
from smt.sampling_methods import LHS
from rombo.rom.nonlinrom import AUTOENCROM
import numpy as np
from rombo.dimensionality_reduction.autoencoder import MLPAutoEnc
from rombo.test_problems.test_problems import EnvModelFunction
from rombo.optimization.rombo import ROMBO

# Importing relevant classes from BoTorch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.models import KroneckerMultiTaskGP, SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

tkwargs = {"device": torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0"), "dtype": torch.float64}
```

Once the necessary modules have been imported, the first step is to instantiate the problem class that the optimization loop is being created for. Here, we use the Environment Model Function class that has already been defined in ROMBO. If a user would like to use ROMBO with a different test problem, it will be necessary to define a problem class which can be done following the code in the `test_problems` folder. Some optimization parameters have also been defined along with the problem instance. It is also assumed that the design variables of the problem have been normalized to the range of 0 to 1. 

```python
# Instantiating the problem and defining optimization parameters
inputdim = 15
outputdim = 1024
xlimits = np.array([[0.0, 1.0]]*inputdim)
n_init = 10
objective = EnvModelFunction(input_dim=inputdim, output_dim=outputdim, normalized=True)
bounds = torch.cat((torch.zeros(1, inputdim), torch.ones(1, inputdim))).to(**tkwargs)
n_iterations = 2
```
The next step is to generate the initial data for the optimization using an LHS sampling plan. 

```python
# Generating the initial sample for the trial
sampler = LHS(xlimits=xlimits, criterion="ese")
xdoe = sampler(n_init)
xdoe = torch.tensor(xdoe, **tkwargs)
ydoe = objective.evaluate(xdoe)
ydoe = ydoe.reshape((ydoe.shape[0], objective.output_dim))
```
After generating the data, we will define the ROM architecture and instantiate the ROMBO optimizer. The `ROMBO` class must be instatitated with the initial data, number of Monte Carlo samples, bounds of the problem, problem class (`MCObjective`), acquisition function and the chosen ROM architecture. 

```python
autoencoder = MLPAutoEnc(high_dim=ydoe.shape[-1], hidden_dims=[256,64], zd = 10, activation = torch.nn.SiLU())
autoencoder.double()
rom_args = {"autoencoder": autoencoder, "low_dim_model": KroneckerMultiTaskGP, "low_dim_likelihood": ExactMarginalLogLikelihood,
            "standard": False, "saas": False}
optim_args = {"q": 1, "num_restarts": 25, "raw_samples": 512}
optimizer1 = ROMBO(init_x=xdoe, init_y=ydoe, num_samples=32, bounds = bounds, MCObjective=objective, acquisition=qLogExpectedImprovement, ROM=AUTOENCROM, ROM_ARGS=rom_args) 
```
Once the `ROMBO` optimizer is initialized, one step of the optimization can simply be done using the `do_one_step` method shown below. The method requires a `tag` which is just a string to label the optimizer while logging the results and `tkwargs` which are the multi-start gradient-based optimization options used while optimizing the acquisition function. This method will train the ROM model, generate the acquisition function, optimize the acquisition function and update the sampling plan provided to the `ROMBO` class with the new infill point. The latest sampling plan of the optimizer can be accesses using `optimizer1.xdoe` and `optimizer1.ydoe`.

```python
optimizer1.do_one_step(tag = 'ROMBO + Log EI', tkwargs=optim_args)
```

If the optimization must be run in a loop for a certain number of iterations, this can be done by including the `do_one_step` method in a simple for loop.

```python
for i in range(n_iterations):
    optimizer1.do_one_step(tag = 'ROMBO + Log EI', tkwargs=optim_args)
```

## Running the example optimization cases for the ROMBO framework

The `examples` folder contains the test cases that were used to characterize the ROMBO framework. Each of the scripts included in the folder also serve as an example of how to use the ROMBO framework to perform optimization. Running the example files as is will reproduce results that are similar to the ones included in the publication for the ROMBO framework. For example, to run the example that utilizes BO and ROMBO for optimizing the Environment Model Function (EMF), ``cd`` into the examples/env_model_function folder and run the following from the terminal:

    python env_model_function_bo.py --input_dim 15 --output_dim 1024 --latent_dim 10 --mc_samples 32 --trial_num 1

This will run the EMF case with the standard BO method and ROMBO method using a latent dimension of 10 and 32 Monte Carlo samples. The options entered in the terminal can be changed to run different trials and variants of the test cases. Other test cases can be run in a similar manner. To find out more about the options for each test case, simply type the following in the terminal after entering the relevant examples folder and replacing `example_script` with the name of any of the scripts present in the folder.  

    python example_script -h

> **_NOTE:_**  Running the airfoil test case requires installing the [blackbox](https://github.com/ComputationalDesignLab/blackbox) package and its dependecies. This is because the computational fluid dynamics solver used in the airfoil case is implemented using that package. 

## Cite this work!

If this framework proves useful for your own work, please cite the following paper


We welcome collaboration on further development of this framework, both theoretically and from a codebase perspective. 

