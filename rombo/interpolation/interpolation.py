"""
PyTorch based models for interpolation procedures in data-driven ROMs

Current Implementation:
    
    - GP models through GPyTorch
    - NN models through PyTorch

"""

# Importing relevant libraries
import torch
import gpytorch
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
    def train(self):
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

# Class definition for a GP model
class GPRModel(MLModel):
    
    """
    Definition of a GP model using the GPyTorch framework for use with ROM models defined in 
    PyROM framework.
    
    """
    
    def __init__(self, model, likelihood, train_x, train_y, tkwargs):
        
        # Initializing model and data

        # This step ensures correct declaration of the likelihood
        if train_y.shape[-1] > 1:
            likelihood = likelihood(num_tasks=train_y.shape[-1])   
        self.model = model(train_x, train_y, likelihood, num_tasks = train_y.shape[-1], kernel = gpytorch.kernels.RBFKernel())
        self.likelihood = likelihood
        self.tkwargs = tkwargs
        self._settraindata(train_x, train_y)
        
    def train(self, num_epochs, verbose):
            
        if torch.cuda.is_available():
            self.model.cuda()
            self.likelihood.cuda()
        
        # Specify number of training iterations
        num_epochs = num_epochs

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        for i in range(num_epochs):
            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            if verbose:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, num_epochs, loss.item()))
            optimizer.step()
    
    def predict(self, test_x, return_format = 'tensor'):
            
        # Set into eval mode
        self.model.eval()
        self.likelihood.eval()
        
        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(test_x))
            mean = predictions.mean
            lower, upper = predictions.confidence_region()
            
        if return_format == "tensor":
            return mean, lower, upper
        
        elif return_format == "numpy":
            return mean.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy()

# Class definition for a GP model built from BoTorch
class BoTorchModel(MLModel):
    
    def __init__(self, model, mll, train_x, train_y, model_args = {}):

        self.model = model
        self.mll = mll
        self._settraindata(train_x, train_y)
        self.model_args = model_args
    
    "Method to train the BoTorch model - this requires significantly less inputs because of preprocessing done by BoTorch"
    def train(self, type = 'mll'):

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
    def train(self):
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

# Class definition for neural network model
class NN(MLModel):

    def __init__(self, model, train_x, train_y, tkwargs):
        
        # Initializing model and data        
        self.model = model
        self._settraindata(train_x, train_y)
        self.tkwargs = tkwargs
            
    def train(self, epochs, verbose):

        if torch.cuda.is_available():
            self.model.cuda()

        # Training neural network model
        loss_function = torch.nn.MSELoss()

        optimizer = torch.optim.Adam(self.model.parameters(),
                                    lr = 1e-3,
                                    weight_decay = 1e-8)
        epochs = epochs
        losses = []
        for epoch in range(epochs):
            
            # Output of Autoencoder
            predictions = self.model(self.train_x)
            
            # Calculating the loss function
            loss = loss_function(predictions, self.train_y)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if verbose:
                print("Epoch:", epoch, "Loss:", loss)
            
            # Storing losses for printing
            losses.append(loss.item())
        
    def predict(self, test_x, return_format = 'tensor'):

        predictions = self.model(test_x)

        if return_format == "tensor":
            return predictions
        
        elif return_format == "numpy":
            return predictions.cpu().numpy()


