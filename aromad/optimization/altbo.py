import torch
import numpy as np
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.optim import optimize_acqf
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
    def gen_init_conditions(self, acqf):

        start_points = gen_batch_initial_conditions(acqf, self.bounds, q = self.optim_args['q'], num_restarts = self.optim_args['num_restarts'],
                                                    raw_samples = self.optim_args['raw_samples'])
        start_points = start_points.reshape((10,13)) 

        # Only returning points where constraints are not violated
        cl_cons, _ = self.cons_model.predict(start_points)
        area_cons = np.vstack([self.MCObjective.area_constraint(x) for x in start_points])
        area_cons = torch.tensor(area_cons, **tkwargs)
        idx = (cl_cons >= self.MCObjective.targetCL) & (area_cons >= 0)
        print(idx)

        return start_points[idx]
    
    "Method to optimize the acquisition function using BoTorch multi-start gradient-based optimization"
    def constrained_optimize_acquistion_torch(self, acqf, bounds, start_points, tkwargs):

        new_point, new_func = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                options={"batch_limit": 5, "maxiter": 200, "init_batch_limit": 5},
                batch_initial_conditions=start_points,
                nonlinear_inequality_constraints=self.constraints,
                **tkwargs
            )

        return new_point, new_func

    "Method to clamp current doe to the feasible set"
    def clamp_to_feasible(self):

        idx = (self.ycons[0] >= self.MCObjective.targetCL) & (self.ycons[1] >= self.MCObjective.base_area)

        return -self.ydoe[idx], self.xdoe[idx.reshape(1,-1)[0], :]
    
    "Defining a function for constraints"
    def lift_cons(self, x):

        return (self.cons_model.predict(x)/self.MCObjective.targetCL) - 1
    
    def area_cons(self, x):
        area_cons = self.MCObjective.area_constraint(x)

        return torch.tensor(area_cons, **tkwargs)

    def do_one_step(self, tag, tkwargs):

        f_clamped, x_clamped = self.clamp_to_feasible()
        self.best_f = f_clamped.max().item()
        self.best_x = f_clamped.argmax().item()
        print("\nBest Objective Value for {}:".format(tag), self.best_f)
        print("Best Design for {}:".format(tag), x_clamped[self.best_x])
        
        # Training the GP model
        obj_model = BoTorchModel(self.gp, self.mll, self.xdoe, self.ydoe, model_args=self.gp_args)
        obj_model.train(type=self.training)

        self.cons_model = BoTorchModel(self.gp, self.mll, self.xcons[0], self.ycons[0], model_args=self.gp_args)
        self.cons_model.train(type=self.training)

        # Creating the acquisition function
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_samples]))
        acqf = self.setacquisition(model=obj_model.model, sampler=sampler, best_f=self.best_f, objective_required = False)
        init_conditions = self.gen_init_conditions(acqf)
        print("Init conditions complete")
        self.constraints = [self.lift_cons, self.area_cons]

        # Optimizing the acquisition function to obtain a new point
        new_x, _ = self.constrained_optimize_acquistion_torch(acqf, self.bounds, init_conditions, tkwargs)

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
