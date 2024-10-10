import torch
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.optim import optimize_acqf
from .basebo import BaseBO
    
class ROMBO(BaseBO):

    "Class definition for ROMBO - utilizes BoTorch to do the calculations and maximization of the acquisition function"

    def __init__(self, init_x, init_y, num_samples, MCObjective, bounds, acquisition, ROM, ROM_ARGS):

        self.xdoe = init_x
        self.ydoe = init_y
        self.bounds = bounds
        self.num_samples = num_samples
        self.acquisition = acquisition
        self.MCObjective = MCObjective
        self.rom = ROM
        self.args = ROM_ARGS

    "Method to set the objective for MC Bayesian optimization"
    def setobjective(self, model):

        "Function definition for MC Objective"
        def function(samples, X=None):

            samples = model.dimensionreduction.backmapping(samples)
            return self.MCObjective.utility(samples)

        self.objective = GenericMCObjective(function)

    "Method to perform only one iteration for running parallel cases with multiple optimizers and multiple settings"
    def do_one_step(self, tag, tkwargs):

        self.best_f = self.MCObjective.utility(self.ydoe).max().item()
        self.best_x = self.MCObjective.utility(self.ydoe).argmax().item()
        print("\nBest Objective Value for {}:".format(tag), self.best_f)
        print("Best Design for {}:".format(tag), self.xdoe[self.best_x])

        rom_model = self.rom(self.xdoe, self.ydoe, **self.args)

        # Training the ROM
        rom_model.trainROM(verbose=False)
        self.setobjective(rom_model)

        # Creating the acquisition function
        sampler = SobolQMCNormalSampler(sample_shape = torch.Size([self.num_samples]))
        acqf = self.setacquisition(model = rom_model.gp_model.model, sampler=sampler, best_f=self.best_f)

        # Optimizing the acquisition function to obtain a new point
        new_x, _ = self.optimize_acquistion_torch(acqf, self.bounds, tkwargs)

        # Add in new data to the existing dataset 
        for x in new_x:
            self.xdoe = torch.cat((self.xdoe, x.unsqueeze(0)), dim = 0)
            new_y = self.MCObjective.function(x)
            new_y = new_y.reshape((1, self.ydoe.shape[-1]))
            self.ydoe = torch.cat((self.ydoe, new_y), dim = 0)

    "Method to run the optimization"
    def optimize(self, tag, n_iterations, tkwargs):

        for iteration in range(n_iterations):

            print("\n\n##### Running iteration {} out of {} #####".format(iteration+1, n_iterations))
            self.do_one_step(tag, tkwargs)


class ConstrainedROMBO(BaseBO):

    "Class definition for ROMBO - utilizes BoTorch to do the calculations and maximization of the acquisition function"

    def __init__(self, init_x, init_y, num_samples, MCObjective, bounds, acquisition, constraints, ROM, ROM_ARGS, OPTIM_ARGS):

        self.xdoe = init_x
        self.ydoe = init_y
        self.bounds = bounds
        self.num_samples = num_samples
        self.acquisition = acquisition
        self.MCObjective = MCObjective
        self.rom = ROM
        self.args = ROM_ARGS
        self.optim_args = OPTIM_ARGS
        self.constraints = constraints

    "Method to generate initial conditions based on acquisition function"
    def gen_init_conditions(self, acqf):

        start_points = gen_batch_initial_conditions(acqf, self.bounds, q = self.optim_args['q'], num_restarts = self.optim_args['num_restarts'],
                                                    raw_samples = self.optim_args['raw_samples'])
        
        # Only returning points where constraints are not violated

        return start_points
    
    "Method to optimize the acquisition function using BoTorch multi-start gradient-based optimization"
    def constrained_optimize_acquistion_torch(self, acqf, bounds, start_points):

        new_point, new_func = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                options={"batch_limit": 5, "maxiter": 200, "init_batch_limit": 5},
                batch_initial_conditions=start_points,
                **self.optim_args
            )

        return new_point, new_func
    
    "Method to set the objective for MC Bayesian optimization"
    def setobjective(self, model):

        "Function definition for MC Objective"
        def function(samples, X=None):

            samples = model.dimensionreduction.backmapping(samples)
            return self.MCObjective.utility(samples)

        self.objective = GenericMCObjective(function)

    "Method to perform only one iteration for running parallel cases with multiple optimizers and multiple settings"
    def do_one_step(self, tag, tkwargs):

        self.best_f = self.MCObjective.utility(self.ydoe).max().item()
        self.best_x = self.MCObjective.utility(self.ydoe).argmax().item()
        print("\nBest Objective Value for {}:".format(tag), self.best_f)
        print("Best Design for {}:".format(tag), self.xdoe[self.best_x])

        rom_model = self.rom(self.xdoe, self.ydoe, **self.args)

        # Training the ROM
        rom_model.trainROM(verbose=False)
        self.setobjective(rom_model)

        # Creating the acquisition function and initial conditions
        sampler = SobolQMCNormalSampler(sample_shape = torch.Size([self.num_samples]))
        acqf = self.setacquisition(model = rom_model.gp_model.model, sampler=sampler, best_f=self.best_f)
        init_conditions = self.gen_init_conditions(acqf)

        # Optimizing the acquisition function to obtain a new point
        new_x, _ = self.constrained_optimize_acquistion_torch(acqf, self.bounds, init_conditions)

        # Add in new data to the existing dataset 
        for x in new_x:
            self.xdoe = torch.cat((self.xdoe, x.unsqueeze(0)), dim = 0)
            new_y = self.MCObjective.function(x)
            new_y = new_y.reshape((1, self.ydoe.shape[-1]))
            self.ydoe = torch.cat((self.ydoe, new_y), dim = 0)

    "Method to run the optimization"
    def optimize(self, tag, n_iterations, tkwargs):

        for iteration in range(n_iterations):

            print("\n\n##### Running iteration {} out of {} #####".format(iteration+1, n_iterations))
            self.do_one_step(tag, tkwargs)


        


