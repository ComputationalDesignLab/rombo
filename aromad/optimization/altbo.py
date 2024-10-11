import torch
import numpy as np
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

        self.xdoe = init_x
        self.ydoe = init_y
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

class AirfoilBO(BaseBO):

    "Class definition for constrained BO with one unknown constraint and one known constraint"
    def __init__(self, obj_x, obj_y, cons_x, cons_y, num_samples, MCObjective, bounds, acquisition, GP, MLL, GP_ARGS = {}, optim_args = {}, training = 'mll'):

        self.xdoe = obj_x
        self.ydoe = obj_y
        self.xcons = cons_x
        self.ycons = cons_y
        self.bounds = bounds
        self.num_samples = num_samples
        self.acquisition = acquisition
        self.MCObjective = MCObjective
        self.gp = GP
        self.mll = MLL
        self.training = training
        self.gp_args = GP_ARGS
        self.optim_args = optim_args

    "Method to generate initial conditions based on acquisition function"
    def gen_init_conditions(self, acqf, bounds):

        start_points = gen_batch_initial_conditions(acqf, bounds, q = self.optim_args['q'], num_restarts = self.optim_args['num_restarts'],
                                                    raw_samples = self.optim_args['raw_samples'])
        start_points = start_points.reshape((self.optim_args['num_restarts'],13)) 

        # Only returning points where constraints are not violated
        cl_cons, _ = self.cons_model.predict(start_points)
        area_cons = np.vstack([self.MCObjective.area_constraint(x.detach().cpu().numpy()) for x in start_points])
        area_cons = torch.tensor(area_cons, **tkwargs)
        idx = (cl_cons >= self.MCObjective.targetCL) & (area_cons >= 0)

        start_points = start_points[idx.reshape(1,-1)[0], :]

        return start_points.reshape((start_points.shape[0], 1, 13))
    
    "Definition of objective function for SLSQP"
    def objective_func(self, x):

        # Reshaping DVs and converting to tensor
        x_reshape = x.reshape((1, x.shape[0]))
        x_tensor = torch.tensor(x_reshape, **tkwargs)

        # Evaluating acquisition function value
        acqf_value = self.acqf(x_tensor)

        return -acqf_value.detach().cpu().numpy()
    
    "Method to optimize the acquisition function using BoTorch multi-start gradient-based optimization"
    def constrained_optimize_acquistion(self, acqf, bounds, start_points):

        result_x = np.array([])
        result_f = np.array([])
        constraints = [NonlinearConstraint(self.lift_cons, -np.inf, 0), NonlinearConstraint(self.area_cons, -np.inf, 0)]
        method = "SLSQP"
        jac = "3-point"
        for start in start_points:
            x0 = start[0].detach().cpu().numpy()
            result = minimize(fun=self.objective_func, x0=x0, method=method, jac=jac, constraints=constraints, bounds=bounds)

            result_x = np.append(result_x, result.x)
            result_f = np.append(result_f, result.f)

        return result_f.max(), result_x[result_f.argmax()]

    "Method to clamp current doe to the feasible set"
    def clamp_to_feasible(self):

        idx = (self.ycons[0] >= self.MCObjective.targetCL) & (self.ycons[1] >= self.MCObjective.base_area)
        return -self.ydoe[idx], self.xdoe[idx.reshape(1,-1)[0], :]
    
    "Defining a function for constraints"
    def lift_cons(self, x):
        x_reshape = x.reshape((1, x.shape[0]))
        x_tensor = torch.tensor(x_reshape, **tkwargs)
        cl, _ = self.cons_model.predict(x_tensor)
        return 1 - (cl.detach().cpu().numpy()/self.MCObjective.targetCL)
    
    def area_cons(self, x):
        area_cons = self.MCObjective.area_constraint(x)
        return area_cons

    def do_one_step(self, tag, cand_bounds):

        f_clamped, x_clamped = self.clamp_to_feasible()
        self.best_f = f_clamped.max().item()
        self.best_x = f_clamped.argmax().item()
        print("\nBest Objective Value for {}:".format(tag), self.best_f)
        print("Best Design for {}:".format(tag), x_clamped[self.best_x])
        
        # Training the GP modelsss
        obj_model = BoTorchModel(self.gp, self.mll, self.xdoe, -self.ydoe, model_args=self.gp_args)
        obj_model.train(type=self.training)

        self.cons_model = BoTorchModel(self.gp, self.mll, self.xcons[0], self.ycons[0], model_args=self.gp_args)
        self.cons_model.train(type=self.training)

        # Creating the acquisition function
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_samples]))
        self.acqf = self.setacquisition(model=obj_model.model, sampler=sampler, best_f=self.best_f, objective_required = False)
        init_conditions = self.gen_init_conditions(self.acqf, cand_bounds)

        # Optimizing the acquisition function to obtain a new point
        new_x, _ = self.constrained_optimize_acquistion(self.acqf, self.bounds, init_conditions)

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
