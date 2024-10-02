import torch
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf
from abc import ABC, abstractmethod

# Pymoo libraries for acqf optimization
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination

class AnalysisFunction:

    "Class definition for evaluation function of the ROMBO algorithm"

    def evaluate(self, function, x):

        """
            Evaluates the function at the given point x
        """
        return function(x)
    
class ROMBO:

    "Class definition for ROMBO - utilizes BoTorch to do the calculations and pymoo to do the maximization"

    def __init__(self, init_x, init_y, num_samples, n_iterations, MCObjective, bounds, acquisition, ROM):

        self.xdoe = init_x
        self.ydoe = init_y
        self.bounds = bounds
        self.num_samples = num_samples
        self.n_iterations = n_iterations
        self.acquisition = acquisition
        self.MCObjective = MCObjective
        self.rom = ROM

    "Method to set the objective for MC Bayesian optimization"
    def setobjective(self, MCObj):

        self.objective = GenericMCObjective(MCObj)

    "Method to optimize the acquisition function - standard is to minimize the function"
    def optimize_acquistion_pymoo(self, optprob):

        "Problem must have the acquistion function as the input"
        problem = optprob(self.acquisition)
        algorithm = DE(pop_size=200, CR=0.9, dither="vector")
        termination = DefaultSingleObjectiveTermination(
            xtol=1e-3,
            cvtol=1e-3,
            ftol=1e-3,
            period=10,
            n_max_gen=1000,
            n_max_evals=200000,
        )
        res = minimize(problem, algorithm, termination = termination, verbose=True)

        return res.X, res.F
    
    "Method to optimize the acquisition function using BoTorch multi-start gradient-based optimization"
    def optimize_acquistion_torch(self, bounds, tkwargs):

        new_point, new_func = optimize_acqf(
                acq_function=self.acquisition,
                bounds=bounds,
                options={"batch_limit": 5, "maxiter": 200, "init_batch_limit": 5},
                **tkwargs
            )

        return new_point, new_func

    "Method to run the optimization"
    def optimize(self, optprob):

        for iteration in range(self.n_iterations):

            print("\n\n##### Running iteration {} out of {} #####".format(iteration+1, self.n_iterations))

            best_f = self.MCObjective(self.ydoe).max()
            best_x = self.MCObjective(self.ydoe).argmax()
            print("Best Objective Value:", best_f.item())
            print("Best Design:", self.xdoe[best_x.item()])

            # Training the ROM
            self.rom.train(verbose=False)

            # Optimizing the acquisition function to obtain a new point
            new_x, _ = self.optimize_acquistion(optprob)

            # Add in new new to the existing dataset 





        


