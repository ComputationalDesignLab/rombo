"""
PyTorch based models for interpolation procedures in data-driven ROMs

Current Implementation:
    
    - GP models through GPyTorch

"""

# Importing relevant libraries
import torch
import gpytorch
from abc import ABC, abstractmethod

# Base class definition for low dimensional model
class MLModel(ABC):

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
            x = torch.tensor(x, **self.tkwargs)
        
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
        self.train_x = self._checkTensor(train_x)
        self.train_y = self._checkTensor(train_y)
        self.tkwargs = tkwargs
        
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

# Class definition for neural network model
class NN(MLModel):

    def __init__(self, model, train_x, train_y, tkwargs):
        
        # Initializing model and data        
        self.model = model
        self.train_x = self._checkTensor(train_x)
        self.train_y = self._checkTensor(train_y)
        self.tkwargs = tkwargs
            
    def train(self, num_epochs, verbose):

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
            
            # Storing losses for printing
            losses.append(loss.item())
        
    def predict(self, test_x, return_format = 'tensor'):

        predictions = self.model(test_x)

        if return_format == "tensor":
            return predictions
        
        elif return_format == "numpy":
            return predictions.cpu().numpy()

