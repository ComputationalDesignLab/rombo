#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:51:36 2023

@author: Abhijnan
"""

"""
Creation of a ROM model using parameters given by a user.

Current Implementation:
    
    - Linear ROM using POD/PCA
    - Nonlinear ROM using Manifold Learning

"""
import torch
import gpytorch

# Setting data type and device for Pytorch based libraries
tkwargs = {
    "dtype": torch.float,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

class LinearROM():
    
    def __init__(self, param_doe, high_dim_data, params):
        
        # Initializing data and parameters of the model
        self.param_doe = param_doe
        self.high_dim_data = high_dim_data
        self.params = params
        

        
        
        
