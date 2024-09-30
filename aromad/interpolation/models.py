"""
Pre-built regression/interpolation models built using PyTorch libraries. These can be directly imported
and used with interpolation models built in the PyROM framework.

Current Implementation:
    
    - Single Task Exact GP Model
    - Multitask Exact GP Model
    - Simple neural network model

"""

# Importing relevant libraries
import gpytorch
import torch.nn as nn

# Class Definition for a MultiTask GP Model 
class MultitaskGPModel(gpytorch.models.ExactGP):
    
    """
    MultiTask GPyTorch Model

    Simple multitask regression model for modelling multiple outputs with some correlation
    
    """
    
    def __init__(self, train_x, train_y, likelihood, num_tasks, kernel):
        
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            kernel, num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
# Class Definition for a Exact GP Model
class ExactGPModel(gpytorch.models.ExactGP):
    
    """
    Exact GP GPyTorch Model

    Simple GP regression model for modelling mapping of inputs to outputs
    
    """
    
    def __init__(self, train_x, train_y, likelihood, kernel, num_tasks = 1):
        
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
# Defining of a simple MLP neural network model
class NNModel(nn.Module):

    """
    Standard NN model with an MLP architecture
    
    """

    def __init__(self, input_dim, hidden_dims, output_dim, activation):
        super(NNModel, self).__init__()
        
        layers = []
        last_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(activation)
            last_dim = dim
        
        layers.append(nn.Linear(last_dim, output_dim))
        
        # Combine layers into a Sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
