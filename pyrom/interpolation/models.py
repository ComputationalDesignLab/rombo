#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:54:21 2023

@author: Abhijnan
"""

"""
Pre-built regression/interpolation models built using PyTorch libraries. These can be directly imported
and used with interpolation models built in the PyROM framework.

Current Implementation:
    
    - Single Task Exact GP Model
    - Multitask Exact GP Model

"""

# Importing relevant libraries
import gpytorch

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
    

