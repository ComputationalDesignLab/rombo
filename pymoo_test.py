import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.optimize import minimize
from pymoo.config import Config
Config.warnings['not_compiled'] = False
nvar = 10
class GriewankRosenbrockFunction(Problem):

    def __init__(self):
        super().__init__(n_var=nvar, n_obj=1, n_constr=0, xl=np.array([-4]*nvar), xu=np.array([4]*nvar))

    def _evaluate(self, x, out, *args, **kwargs):

        n_samples, n_vars = x.shape
        f_values = np.zeros(n_samples)
        
        for i in range(n_samples):
            X = x[i, :]
            f_values[i] = self._rosenbrock_griewank(X)
        
        out["F"] = f_values.reshape(-1, 1)

    def _rosenbrock_griewank(self, x):
        n = len(x)
        
        if n < 2:
            raise ValueError("Rosenbrock-Griewank function requires at least 2 dimensions")
        
        # Rosenbrock part
        _sum = 0.0
        
        for i in range(n - 1):
            # Rosenbrock terms
            term1 = 100 * (x[i+1] - x[i]**2)**2
            term2 = (1 - x[i])**2
            _sum += (term1 + term2)/4000 - np.cos(term1 + term2)
        
        # Combine Rosenbrock and Griewank components
        f_value = (10/(n-1)) * _sum + 10
        
        return f_value

problem = GriewankRosenbrockFunction()
print(problem.evaluate(np.array([[-1.0]*problem.n_var])))
pop_size = 20#100 * problem.n_var # Number of individuals in the population: 10 times number of variables
sampling = LHS() # How the initial population is sampled

algorithm = DE(pop_size=pop_size, variant="DE/best/1/bin",
                CR=0.9, F=0.5, dither="vector")

termination = DefaultSingleObjectiveTermination(
    xtol=1e-6,
    cvtol=1e-6,
    ftol=1e-6,
    period=10,
    n_max_gen=1000,
    n_max_evals=100000
)

res = minimize(problem, algorithm, termination=termination, verbose=True, 
               save_history=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
