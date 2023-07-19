#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:58:08 2023

@author: Abhijnan
"""

"""
PyTorch based models for interpolation procedures in data-driven ROMs

Current Implementation:
    
    - GP models through GPyTorch

"""

# Importing relevant libraries
import torch
import gpytorch

# Class definition for a ROM GP model
class GPyTorchModel():
    
    """
    Definition of a GP model using the GPyTorch framework for use with ROM models defined in 
    PyROM framework.
    
    """
    
    def __init__(self, model, likelihood, train_x, train_y, tkwargs):
        
        # Initializing model and data        
        self.model = model
        self.likelihood = likelihood
        self.train_x = self._checkTensor(train_x)
        self.train_y = self._checkTensor(train_y)
        self.tkwargs = tkwargs
        
    def train(self, num_epochs, verbose):
            
        if torch.cuda.is_available():
            model = self.model.cuda()
            likelihood = self.likelihood.cuda()
        
        # Specify number of training iterations
        num_epochs = num_epochs

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        print("########Training GPyTorch Model")
        for i in range(num_epochs):
            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            if verbose:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, num_epochs, loss.item()))
            optimizer.step()
        
        return model, likelihood
    
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

    def _checkTensor(self, x):
        
        "Method to check whether a given variable is a Tensor or not"
        
        if not torch.is_tensor(x):
            x = torch.tensor(x, **self.tkwargs)
        
        return x
            
            
            
        


