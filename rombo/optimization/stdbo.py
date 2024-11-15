import torch
import numpy as np
from functools import reduce
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim.initializers import gen_batch_initial_conditions
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from .basebo import BaseBO
from ..interpolation.interpolation import BoTorchModel

# Setting data type and device for Pytorch based libraries
tkwargs = {
    "dtype": torch.float,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

class BO(BaseBO):

    "Class definition for BayesOpt - this can be used to perform Bayesian optimization using single GP models"

    def __init__(self, init_x, init_y, num_samples, MCObjective, bounds, acquisition, GP, MLL, GP_ARGS = {}, training = 'mll'):

        self.xdoe = self._checkTensor(init_x)
        self.ydoe = self._checkTensor(init_y)
        self.bounds = bounds
        self.num_samples = num_samples
        self.acquisition = acquisition
        self.MCObjective = MCObjective
        self.gp = GP
        self.mll = MLL
        self.training = training
        self.gp_args = GP_ARGS

    def do_one_step(self, tag, tkwargs):

        self.best_f = self.ydoe.max().item()
        self.best_x = self.ydoe.argmax().item()
        print("\nBest Objective Value for {}:".format(tag), self.best_f)
        print("Best Design for {}:".format(tag), self.xdoe[self.best_x])
        
        # Training the GP model
        gp_model = BoTorchModel(self.gp, self.mll, self.xdoe, self.ydoe, model_args=self.gp_args)
        gp_model.train(type=self.training)

        # Creating the acquisition function
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_samples]))
        acqf = self.setacquisition(model=gp_model.model, sampler=sampler, best_f=self.best_f, objective_required = False)

        # Optimizing the acquisition function to obtain a new point
        new_x, _ = self.optimize_acquistion_torch(acqf, self.bounds, tkwargs)

        if self.training == 'bayesian':
            self.lengthscales = gp_model.model.median_lengthscale.detach().cpu().numpy()

        # Add in new data to the existing dataset 
        for x in new_x:
            self.xdoe = torch.cat((self.xdoe, x.unsqueeze(0)), dim = 0)
            new_y = self.MCObjective.function(x)
            new_score = self.MCObjective.utility(new_y)
            self.ydoe = torch.cat((self.ydoe, new_score.reshape((1,self.ydoe.shape[-1]))), dim = 0)

    "Method to run the optimization in a loop"
    def optimize(self, tag, n_iterations, tkwargs):

        for iteration in range(n_iterations):

            print("\n\n##### Running iteration {} out of {} #####".format(iteration+1, n_iterations))
            self.do_one_step(tag, tkwargs)

class ConstrainedBO(BaseBO):

    "Class definition for constrained BO with known and unknown constraints"
    def __init__(self, obj_x, obj_y, cons_x, cons_y, cons_limit, n_unknown_cons, cons_known, num_samples, MCObjective, lowerBounds, upperBounds, 
                 acquisition, GP, MLL, GP_ARGS = {}, optim_args = {}, training = 'mll'):

        self.xdoe = obj_x
        self.ydoe = obj_y
        self.xcons = cons_x
        self.ycons = cons_y
        self.constraint_limits = cons_limit
        self.n_unknown_cons = n_unknown_cons
        self.cons_known = cons_known
        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds
        self.num_samples = num_samples
        self.acquisition = acquisition
        self.MCObjective = MCObjective
        self.gp = GP
        self.mll = MLL
        self.training = training
        self.gp_args = GP_ARGS
        self.optim_args = optim_args
    
    "Definition of objective function for SLSQP"
    def objective_func(self, x):

        # Reshaping DVs and converting to tensor
        x_tensor = torch.tensor(x.reshape((1,x.shape[0])), **tkwargs)

        # Evaluating acquisition function value
        acqf_value = self.acqf(x_tensor)

        return -acqf_value.item()

    "Method to clamp current doe to the feasible set"
    def clamp_to_feasible(self):

        ydoe_prime = self.ydoe.copy()
        xdoe_prime = self.xdoe.copy()

        idx_list = []
        for i in range(len(self.constraint_limits)):

            idx = np.where(self.ycons[i] >= self.constraint_limits[i])
            idx_list.append(idx)

        idx = reduce(np.intersect1d,(idx_list[0][0], idx_list[1][0])) # Need to generalize this statement
        return -ydoe_prime[idx], xdoe_prime[idx.reshape(1,-1)[0], :]
    
    "Method to generate and train a GP model for constraints or objectives"
    def _train_model(self, xdata, ydata):

        gp_model = BoTorchModel(self.gp, self.mll, xdata, ydata, model_args=self.gp_args)
        gp_model.train(type=self.training)

        return gp_model
    
    "Method to generate a function for each of the constraints"
    def _generate_constraint(self, cons_model):

        cons_fun = lambda x: cons_model.predict(torch.tensor(x.reshape((1,x.shape[0])), **tkwargs), return_format = "numpy")
        return cons_fun

    def do_one_step(self, tag):

        f_clamped, x_clamped = self.clamp_to_feasible()
        self.best_f = f_clamped.max().item()
        self.best_x = x_clamped[f_clamped.argmax().item()]
        print("\nBest Objective Value for {}:".format(tag), self.best_f)
        print("Best Design for {}:".format(tag), self.best_x)
        
        # Training the GP models for objective function
        obj_model = self._train_model(self.xdoe, -self.ydoe)
        
        # Training the GP models fo necessary constraints - the unknown constraints are specified first in the list
        cons_model_list = []
        cons_func_list = []
        for i in range(self.n_unknown_cons):
            cons_model = self._train_model(self.xcons[i], self.ycons[i])
            cons_model_list.append(cons_model)
            cons_func_list.append(self._generate_constraint(cons_model))

        [cons_func_list.append(self.cons_known[i]) for i in range(len(self.cons_known))]

        # Creating the acquisition function
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_samples]))
        self.acqf = self.setacquisition(model=obj_model.model, sampler=sampler, best_f=self.best_f, objective_required = False)

        # Optimizing the acquisition function to obtain a new point
        new_x, _ = self.optimize_acquistion_pymoo(self.objective_func, lowerBounds=self.lowerBounds, upperBounds=self.upperBounds,
                                                  cons_func_list=cons_func_list)

        # Add in new data to the existing dataset 
        # for x in new_x:
        #     self.xdoe = torch.cat((self.xdoe, x.unsqueeze(0)), dim = 0)
        #     new_y = self.MCObjective.function(x)
        #     new_score = self.MCObjective.utility(new_y)
        #     self.ydoe = torch.cat((self.ydoe, new_score.reshape((1,self.ydoe.shape[-1]))), dim = 0)

    "Method to run the optimization in a loop"
    def optimize(self, tag, n_iterations, tkwargs):

        for iteration in range(n_iterations):

            print("\n\n##### Running iteration {} out of {} #####".format(iteration+1, n_iterations))
            self.do_one_step(tag, tkwargs)
