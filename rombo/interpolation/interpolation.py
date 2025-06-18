# Some of the code in this script has been adapted from the BoTorch code and tutorials
# https://botorch.org/docs/tutorials/, https://botorch.readthedocs.io/en/stable/

# For more details on the implementation of the deep kernel learning model, see the following papers and repositories
# 1. N. Maus, Z. J. Lin, M. Balandat, E. Bakshy, Joint composite latent
# space bayesian optimization, in: Forty-first International Conference
# on Machine Learning, Vienna, Austria, 21-27 July, 2024. Code : https://github.com/nataliemaus/joco_icml24

# 2. GPyTorch Documentation: https://docs.gpytorch.ai/en/v1.13/examples/06_PyTorch_NN_Integration_DKL/KISSGP_Deep_Kernel_Regression_CUDA.html

# Importing relevant libraries
import torch
import torch.nn as nn
import gpytorch
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.posteriors.gpytorch import GPyTorchPosterior
from abc import ABC, abstractmethod

# Arguments for GPU-related calculations
tkwargs = {"device": torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0"), "dtype": torch.float64}

# Base class definition for low dimensional model
class MLModel(ABC):

    "Method to set the training data for the model"
    def _settraindata(self, train_x, train_y):

        self.train_x = self._checkTensor(train_x)
        self.train_y = self._checkTensor(train_y)

    "Method to train the ML model for the low dimensional space"
    @abstractmethod
    def trainModel(self):
        pass

    "Method to predict the low dimensional space using the ML model"
    @abstractmethod
    def predict(self):
        pass

    "Method to check whether a given variable is a Tensor or not - useful for PyTorch based models"
    def _checkTensor(self, x):
        
        if not torch.is_tensor(x):
            x = torch.tensor(x, **tkwargs)
        
        return x

# Class definition for a GP model built from BoTorch
class BoTorchModel(MLModel):
    
    def __init__(self, model, mll, train_x, train_y, model_args = {}):

        self.model = model
        self.mll = mll
        self._settraindata(train_x, train_y)
        self.model_args = model_args
    
    "Method to train the BoTorch model - this requires significantly less inputs because of preprocessing done by BoTorch"
    def trainModel(self, type = 'mll'):

        # Instantiate the model
        gp = self.model(self.train_x, self.train_y.to(**tkwargs), **self.model_args)

        # Train model
        if type == 'mll':
            gp = self.fit_mll(gp)
        elif type == 'bayesian':
            gp = self.fit_bayesian(gp)
        
        self.model = gp

    "Method to train using maximum likelihood estimation"
    def fit_mll(self, gp):

        likelihood = self.mll(gp.likelihood, gp)
        fit_gpytorch_mll_torch(likelihood)

        return gp

    "Method to use fully Bayesian training and No U-Turn sampling"
    def fit_bayesian(self, gp, WARMUP_STEPS=256, NUM_SAMPLES=128, THINNING=16):

        fit_fully_bayesian_model_nuts(gp, warmup_steps=WARMUP_STEPS, num_samples=NUM_SAMPLES, thinning=THINNING, disable_progbar=True)

        return gp

    "Method to predict using the BoTorch model"
    def predict(self, xtest, return_format = 'tensor'):

        # Obtaining the posterior distribution of the model
        posterior = self.model.posterior(xtest)
        # Obtaining mean value predictions
        predictions = posterior.mean
        # Obtaining variances
        variances = posterior.variance

        if return_format == "tensor":
            return predictions, variances
        
        elif return_format == "numpy":
            return predictions.cpu().numpy(), variances.cpu().numpy()

# Class definition for a GP model list built from BoTorch
class BoTorchModelList(MLModel):
    
    def __init__(self, model, mll, train_x, train_y, model_args = {}):

        self.model = model
        self.mll = mll
        self._settraindata(train_x, train_y)
        self.model_args = model_args
    
    "Method to train the BoTorch model - this requires significantly less inputs because of preprocessing done by BoTorch"
    def trainModel(self):
        models = []
        for i in range(self.train_y.shape[-1]):
            train_Y = self.train_y[..., i : i + 1]
            models.append(self.model(self.train_x, train_Y, **self.model_args))
        gp = ModelListGP(*models)

        # Train model
        gp = self.fit_mll(gp)
        
        self.model = gp

    "Method to train using maximum likelihood estimation"
    def fit_mll(self, gp):

        likelihood = self.mll(gp.likelihood, gp)
        fit_gpytorch_mll_torch(likelihood)

        return gp

    "Method to predict using the BoTorch model list"
    def predict(self, xtest, return_format = 'tensor'):

        # Obtaining the posterior distribution of the model
        posterior = self.model.posterior(xtest)
        # Obtaining mean value predictions
        predictions = posterior.mean
        # Obtaining variances
        variances = posterior.variance

        if return_format == "tensor":
            return predictions, variances
        
        elif return_format == "numpy":
            return predictions.cpu().numpy(), variances.cpu().numpy()

# Definition of the deep kernel learning model
# This definition is slightly different from the BoTorch models since the model
# is being built from scratch rather than directly being imported from BoTorch
class DeepKernelGP(gpytorch.models.ExactGP, MLModel):

    # Need to standardize the output data and add in that capability
    def __init__(self, train_x, train_y, hidden_dims, zd=2,activation=nn.SiLU()):

        # Specifying the likelihood as the Gaussian likelihood
        super(DeepKernelGP, self).__init__(train_x, train_y, gpytorch.likelihoods.GaussianLikelihood())
        self._settraindata(train_x, train_y)
        
        # Setting the mean and kernel function for the GP model
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.num_outputs = 1

        # Setting up the feature extractor
        encoder_layers = []
        last_dim = train_x.shape[-1]

        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(last_dim, dim))
            encoder_layers.append(activation)
            last_dim = dim
        
        encoder_layers.append(nn.Linear(last_dim, zd))
        self.feature_extractor = nn.Sequential(*encoder_layers)

    def forward(self, x):

        projected_x = self.feature_extractor(x) # Projecting the inputs to the latent space
        mean_x = self.mean_module(projected_x) 
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def posterior(self, X, output_indices=None, observation_noise=False, *args, **kwargs) -> GPyTorchPosterior:

        self.eval()
        self.likelihood.eval()
        dist = self.likelihood(self(X))
        return GPyTorchPosterior(distribution=dist)
    
    def trainModel(self):

        self.train().cuda().double() # Set the model in training mode
        self.likelihood.train().cuda().double()

        optimizer = torch.optim.Adam([
            {'params': self.feature_extractor.parameters()},
            {'params': self.covar_module.parameters()},
            {'params': self.mean_module.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=0.01)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for _ in range(1000):
            optimizer.zero_grad()
            output = self(self.train_x)
            loss = -mll(output, self.train_y)
            print(loss)
            loss.backward()
            optimizer.step()

    def predict(self, xtest, return_format = 'tensor'):

        # Obtaining the posterior distribution of the model
        posterior = self.posterior(xtest)
        # Obtaining mean value predictions
        predictions = posterior.mean
        # Obtaining variances
        variances = posterior.variance

        if return_format == "tensor":
            return predictions, variances
        
        elif return_format == "numpy":
            return predictions.cpu().numpy(), variances.cpu().numpy()



