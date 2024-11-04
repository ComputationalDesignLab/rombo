import numpy as np
from botorch.optim import optimize_acqf
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
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
    
    "Method to optimize the acquisition function using pymoo and differential evolution"
    def optimize_acquistion_pymoo(self, objective_func, lowerBounds, upperBounds, cons_func_list):

        class OptProb(ElementwiseProblem):

            def __init__(self):
                super().__init__(n_var=13, n_obj=1, n_ieq_constr=len(cons_func_list), xl=lowerBounds, xu=upperBounds)

            def _evaluate(self, x, out, *args, **kwargs):

                out["F"] = objective_func(x)
                out["G"] = np.column_stack([cons_func_list[i](x) for i in range(len(cons_func_list))])

        problem = OptProb()
        algorithm = DE(pop_size=200, CR=0.8, dither="vector")
        termination = DefaultSingleObjectiveTermination(xtol=1e-3,cvtol=1e-3,ftol=1e-3,period=10,n_max_gen=1000,n_max_evals=200000)
        res = minimize(problem, algorithm, termination = termination, verbose=False)

        return res.X, res.F
    
    "Method to set the acquisition function for the optimization problem"
    def setacquisition(self, model, sampler, best_f, objective_required = True):

        if objective_required:
            acqf = self.acquisition(model = model, best_f=best_f, sampler=sampler, objective=self.objective)
        else:
            acqf = self.acquisition(model = model, best_f=best_f, sampler=sampler)

        return acqf
    
    @abstractmethod
    def objective_func(self):
        pass

    @abstractmethod
    def setobjective(self):
        pass

    @abstractmethod
    def do_one_step(self, tag, tkwargs):
        pass

    @abstractmethod
    def optimize(self, n_iterations, tkwargs):
        pass
