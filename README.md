# ROMBO: A composite Bayesian optimization framework for engineering design using nonintrusive reduced-order models

ROMBO is a optimization framework that utilizes a composite function formulation and nonlinear nonintrusive reduced order models. An autoencoder neural network is used to 
project high dimensional outputs into a latent space. The latent space is modeled using multi-task Gaussian process models that utilize a Kronecker structure or intrinsic model coregionalization (ICM) formulation. The framework uses a Monte Carlo expected improvement infill strategy to balance exploration of the design space with exploitation of the objective function. A linear POD method is also implemented using the same structure as ROMBO but using POD for dimensionality reduction and independent GP models for the latent space. A standard BO implementation is also provided for generating comparison data for the ROMBO framework. The framework is built utilizing PyTorch and associated libraries such as GPyTorch and BoTorch. Modular base classes have been provided for users to implement their own ROM architectures and utilize them within this framework.

<p align="center">
<img src="images/rombo-1.png"/>
</p>

### Training a simple nonintrusive reduced order model within the ROMBO framework

The following example code demonstrates how the ROMBO framework modules can be used to define a deep learning ROM model and train it to predict the environment model function using
the corresponding test problem class. First, the relevant modules and libraries are imported. The SMT package is used to generate a latin hypercube sampling (LHS) plan and the samples are
evaluated using the evaluate method of the test problem class. A set of testing data is also generated in a similar manner. The autoencoder architecture is defined using the pre-built model
inside ROMBO that allows the creation of a simple fully-connected autoencoder network. A user may also define their own architecture using PyTorch and use it along with the ROM model class within the ROMBO framework. After defining the autoencoder, the `AUTOENCROM` class can be used to define a ROM model with the corresponding training data and GP model. The GP model is imported from the BoTorch package which contains GP models built using GPyTorch.  

```python
import torch 
from smt.sampling_methods import LHS
from rombo.rom.nonlinrom import AUTOENCROM
import numpy as np
from rombo.dimensionality_reduction.autoencoder import MLPAutoEnc
from rombo.test_problems.test_problems import EnvModelFunction
from rombo.optimization.rombo import ROMBO
from botorch.models import KroneckerMultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

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

# Generating the nonlinear ROM model
autoencoder = MLPAutoEnc(high_dim=problem.output_dim, hidden_dims=[256,64], zd = 10, activation = torch.nn.SiLU())
rom = AUTOENCROM(xtrain, htrain, autoencoder = autoencoder, low_dim_model = KroneckerMultiTaskGP, low_dim_likelihood = ExactMarginalLogLikelihood)

# Training the ROM and predicting on the test data
rom.trainROM(verbose=False)
field = rom.predictROM(xtest)
```

