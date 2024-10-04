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
    def setobjective(self):

        def function(samples, X=None):

            samples = self.rom.dimensionreduction.backmapping(samples)
            
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
    def setacquisition(self, sampler, best_f):

        acqf = self.acquisition(model = self.rom.gp_model, best_f=best_f, sampler=sampler, objective=self.objective)

        return acqf

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

            # Creating the acquisition function
            sampler = SobolQMCNormalSampler(sample_shape = torch.Size([self.num_samples]))
            acqf = self.setacquisition(sampler=sampler, best_f=best_f)

            # Optimizing the acquisition function to obtain a new point
            new_x, _ = self.optimize_acquistion(acqf, self.bounds, tkwargs)

            # Add in new data to the existing dataset 
            self.xdoe = torch.cat((self.xdoe, new_x), dim = 0)
            new_y = self.MCObjective.function(new_x)
            new_y = new_y.reshape((1, self.ydoe.shape[-1]))
            self.ydoe = torch.cat((self.ydoe, new_y), dim = 0)




        


