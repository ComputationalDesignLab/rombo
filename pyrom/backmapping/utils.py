#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:19:44 2023

@author: Abhijnan
"""

"""
Backmapping utilities for creating of data-driven ROMs. 

Backmapping enables the reconstruction of a solution in the high dimensional space from 
the solution in the low dimensional space predicted by the interpolation model

Current implementation:
    
    - Linear backmapping
    - Non-linear backmapping via optimization
    
"""

# Importing relevant libraries
import torch
import numpy as np
from math import dist
from scipy.optimize import minimize, NonlinearConstraint, differential_evolution

# Backmapping for a linear ROM that uses POD/PCA methods
def lin_backmap(phi, a):
    
    if torch.is_tensor(a): # Checking if input is a tensor
        
        if a.device.type == 'cuda': # Shifting from GPU to CPU if required
            a = a.cpu()
            
        a = a.numpy()
        a = a.reshape((a.shape[0]))
    
    field = phi @ a.T; # Reconstruction is a simple linear combination in the case of linear methods
    
    return field

# Utilities for nonlinear backmapping

# Definition of k-nearest neighbour calculation based on Euclidean Distance
def k_nearest_neighbours(p, manifold, num_neighbours, train = True):
    
    #p = p.cpu().numpy()
    if train:
        distance = np.zeros(manifold.shape[0] - 1)
    else:
        distance = np.zeros(manifold.shape[0])
    i = 0
    for point in manifold:
        p = p.reshape(-1)
        if np.array_equal(p, point):
            continue
        else:
            distance[i] = dist(p, point)
            i += 1
        
    sorted_distance = np.sort(distance)[:num_neighbours]
    neighbours = np.zeros((num_neighbours, manifold.shape[1]))
    
    j = 0
    for d in sorted_distance:
            
            index = np.where(distance == d)
            neighbours[j,:] = manifold[index,:]   
            j += 1
    
    return neighbours

# Function to calculate regularization parameter
def regularization(p, neighbours, eps, gamma):
    
    # Calculating max L2 norm between point and neighbors
    max_L2 = 0
    for neighbour in neighbours:
        
        L2 = np.linalg.norm(p - neighbour,2)
        
        if L2 > max_L2:
            max_L2 = L2
    
    cj = np.zeros(len(neighbours))
    # Calculating regularization parameter
    i = 0
    for neighbour in neighbours:
        
        cj[i] = eps * ((np.linalg.norm(p - neighbour,2))/max_L2) ** gamma
        i += 1
        
    return cj

# Objective function for backmapping
def obj_func(w, p, neighbours, eps, gamma):
    
    lin_combo = 0
    for i in range(len(neighbours)):
        lin_combo += w[i] * neighbours[i]

    first_term = np.linalg.norm(p - lin_combo, 2) ** 2
    cj = regularization(p, neighbours, eps = eps, gamma = gamma)
    
    second_term = 0
    for i in range(len(cj)):
        second_term += cj[i] * (w[i] ** 2)
    
    return first_term + second_term
    
# Constraint function for backmapping
def constraint(w):
    
    return np.sum(w) - 1

g = NonlinearConstraint(constraint, lb = '-inf', ub = 0)

# Backmapping for a non-linear ROM that uses Isomap/LLE/MDS methods
def nonlin_backmap(low_dim_coords, low_dim_space, n_neighbours, pressure_data, eps, gamma, train):
    
    neighbours = k_nearest_neighbours(low_dim_coords, low_dim_space, num_neighbours = n_neighbours, train = train)
    
    # Setting up and solving optimization problem for backmapping
    w0 = 0.5*np.ones(len(neighbours))
    #low_dim_coords = low_dim_coords.cpu().numpy()
    args = (low_dim_coords, neighbours, eps, gamma)
    method = "SLSQP"
    jac = "2-point"
    bounds = [(0.0,1.0)]*len(neighbours)
    options = {"disp": True}
    
    res = minimize(obj_func, w0, bounds = bounds, args = args, method = method, jac = jac, constraints = g, options = options)
    w_star = res.x # Storing optimal value of weights for linear combination
    print(res.jac)
    
    # Taking linear combination of high dimensional data
    W = 0
    k = 0
    for neighbour in neighbours:
        
        index = np.where(low_dim_space == neighbours[0])
        W += w_star[k] * pressure_data[index[0][0], :]
        k += 1
    
    return W, w_star

# Backmapping for a non-linear ROM using differential evolution
def nonlin_backmap_DE(low_dim_coords, low_dim_space, n_neighbours, pressure_data, eps, gamma, train):
    
    neighbours = k_nearest_neighbours(low_dim_coords, low_dim_space, num_neighbours = n_neighbours, train = train)
    
    # Calculating bounds
    bounds = [(0.0,1.0)]*len(neighbours)
        
    #low_dim_coords = low_dim_coords.cpu().numpy()
    args = (low_dim_coords, neighbours, eps, gamma)
    
    result = differential_evolution(obj_func, bounds = bounds, args = args, constraints = g, disp = True)
    
    w_star = result.x # Storing optimal value of weights for linear combination
    
    # Taking linear combination of high dimensional data
    W = 0
    k = 0
    for neighbour in neighbours:
        
        index = np.where(low_dim_space == neighbour)
        W += w_star[k] * pressure_data[index[0][0], :]
        k += 1
        
    return W, w_star


