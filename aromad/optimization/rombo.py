import torch
from botorch.acquisition.objective import GenericMCObjective
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from abc import ABC, abstractmethod

class AnalysisFunction:

    "Class definition for evaluation function of the ROMBO algorithm"

    def evaluate(self, function, x):

        """
            Evaluates the function at the given point x
        """
        return function(x)
    
class ROMBO:

    "Class definition for ROMBO - utilizes BoTorch to do the calculations and pymoo to do the maximization"

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
    def setobjective(self, rom_model):

        def function(samples, X=None):

            samples = rom_model.dimensionreduction.backmapping(samples)
            return self.MCObjective.utility(samples)

        self.objective = GenericMCObjective(function)
    
    "Method to optimize the acquisition function using BoTorch multi-start gradient-based optimization"
    def optimize_acquistion_torch(self, acqf, bounds, tkwargs):

        new_point, new_func = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                options={"batch_limit": 5, "maxiter": 200, "init_batch_limit": 5},
                **tkwargs
            )

        return new_point, new_func
    
    "Method to set the acquisition function for the optimization problem"
    def setacquisition(self, rom, sampler, best_f):

        acqf = self.acquisition(model = rom.gp_model.model, best_f=best_f, sampler=sampler, objective=self.objective)

        return acqf
    
    "Method to perform only one iteration for running parallel cases with multiple optimizers and multiple settings"
    def do_one_step(self, tag, tkwargs):

        best_f = self.MCObjective.utility(self.ydoe).max()
        best_x = self.MCObjective.utility(self.ydoe).argmax()
        print("\nBest Objective Value for {}:".format(tag), best_f.item())
        print("Best Design for {}:".format(tag), self.xdoe[best_x.item()])

        rom_model = self.rom(self.xdoe, self.ydoe, **self.args)

        # Training the ROM
        rom_model.trainROM(verbose=False)
        self.setobjective(rom_model)

        # Creating the acquisition function
        sampler = SobolQMCNormalSampler(sample_shape = torch.Size([self.num_samples]))
        acqf = self.setacquisition(rom = rom_model, sampler=sampler, best_f=best_f)

        # Optimizing the acquisition function to obtain a new point
        new_x, _ = self.optimize_acquistion_torch(acqf, self.bounds, tkwargs)

        # Add in new data to the existing dataset 
        for x in new_x:
            self.xdoe = torch.cat((self.xdoe, x.unsqueeze(0)), dim = 0)
            new_y = self.MCObjective.function(x)
            new_y = new_y.reshape((1, self.ydoe.shape[-1]))
            self.ydoe = torch.cat((self.ydoe, new_y), dim = 0)

    "Method to run the optimization"
    def optimize(self, n_iterations, tkwargs):

        for iteration in range(n_iterations):

            print("\n\n##### Running iteration {} out of {} #####".format(iteration+1, n_iterations))

            best_f = self.MCObjective.utility(self.ydoe).max()
            best_x = self.MCObjective.utility(self.ydoe).argmax()
            print("Best Objective Value:", best_f.item())
            print("Best Design:", self.xdoe[best_x.item()])

            rom_model = self.rom(self.xdoe, self.ydoe, **self.args)

            # Training the ROM
            rom_model.trainROM(verbose=False)
            self.setobjective(rom_model)

            # Creating the acquisition function
            sampler = SobolQMCNormalSampler(sample_shape = torch.Size([self.num_samples]))
            acqf = self.setacquisition(rom = rom_model, sampler=sampler, best_f=best_f)

            # Optimizing the acquisition function to obtain a new point
            new_x, _ = self.optimize_acquistion_torch(acqf, self.bounds, tkwargs)

            # Add in new data to the existing dataset 
            for x in new_x:
                self.xdoe = torch.cat((self.xdoe, x.unsqueeze(0)), dim = 0)
                new_y = self.MCObjective.function(x)
                new_y = new_y.reshape((1, self.ydoe.shape[-1]))
                self.ydoe = torch.cat((self.ydoe, new_y), dim = 0)




        


