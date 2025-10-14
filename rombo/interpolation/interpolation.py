# Some of the code in this script has been adapted from the BoTorch code and tutorials
# https://botorch.org/docs/tutorials/, https://botorch.readthedocs.io/en/stable/

# Importing relevant libraries
import torch
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.models.model_list_gp_regression import ModelListGP
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
    # To replicate SAAS settings for the airfoil case shown in the publication, the warmup steps, the number of samples were significantly 
    # lowered to enable a faster runtime on a computing cluster without GPU acceleration. 
    # To have settings and performance that are similar to the publication, you can set the following: WARMUP_STEPS=128, NUM_SAMPLES=16, THINNING=16
    # To compare with the exact performance for the airfoil case, it will be more prudent to use the data given directly in the final_results.zip folder.
    
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




