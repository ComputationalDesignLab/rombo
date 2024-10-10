from botorch.optim import optimize_acqf
from abc import ABC, abstractmethod

class BaseBO:

    "Class definition for the base class for performing Bayesian optimization - built using BoTorch"
    
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
    def setacquisition(self, model, sampler, best_f, objective_required = True):

        if objective_required:
            acqf = self.acquisition(model = model, best_f=best_f, sampler=sampler, objective=self.objective)
        else:
            acqf = self.acquisition(model = model, best_f=best_f, sampler=sampler)

        return acqf

    @abstractmethod
    def setobjective(self):
        pass

    @abstractmethod
    def do_one_step(self, tag, tkwargs):
        pass

    @abstractmethod
    def optimize(self, n_iterations, tkwargs):
        pass
