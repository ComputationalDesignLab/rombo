# Some of the code used in this test problems script has been adapted from the GitHub repositories of the following research papers

# 1. N. Maus, Z. J. Lin, M. Balandat, E. Bakshy, Joint composite latent
# space bayesian optimization, in: Forty-first International Conference
# on Machine Learning, Vienna, Austria, 21-27 July, 2024. Code : https://github.com/nataliemaus/joco_icml24

# 2. M. A. Bhouri, M. Joly, R. Yu, S. Sarkar, P. Perdikaris, Scalable bayesian optimization with randomized prior networks, 
# Computer Methods in Applied Mechanics and Engineering 417 (12 2023).
# doi:10.1016/j.cma.2023.116428. Code: https://github.com/bhouri0412/rpn_bo

from .baseproblem import TestFunction
import os
import torch
import numpy as np
from scipy.io import loadmat
import math
import matplotlib.pyplot as plt 
from pde import PDE, FieldCollection, ScalarField, UnitGrid

# Arguments for GPU-related calculations
tkwargs = {"device": torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0"), "dtype": torch.float64}

class RosenbrockFunction(TestFunction):

    def __init__(self, input_dim=10, output_dim=18, normalized = False):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalized = normalized

        self.lower_bounds = [-4]*input_dim
        self.upper_bounds = [4]*input_dim

    def function(self, x):
        if self.normalized:
            xnew = x.clone()
            for i in range(self.input_dim):
                xnew[i] = (
                    xnew[i] * (self.upper_bounds[i] - self.lower_bounds[i])
                ) + self.lower_bounds[i]
        else:
            xnew = x

        x_first = xnew[0:-1]
        x_sq = x_first**2
        x_next = xnew[1:]
        diffs = x_next - x_sq
        h_x = torch.cat((diffs, x_first))
        h_x = h_x.reshape(1, -1)

        return h_x

    def evaluate(self, X):
        return torch.stack([self.function(x) for x in X])

    def score(self, y):
        
        y_first = y[0 : self.input_dim - 1]
        term1 = 100 * (y_first**2)
        y_next = y[self.input_dim - 1 :]
        term2 = (y_next - 1) ** 2
        rosen = term1 + term2
        griewank = (rosen/4000 - torch.cos(rosen))
        reward = (10/(self.input_dim-1))*griewank.sum() + 10
        reward = reward * -1
        return reward

    def utility(self, Y):
        if Y.dim() > 2:
            Y = Y.reshape(Y.shape[0], Y.shape[1], -1)
            return torch.stack([torch.stack([self.score(yprime) for yprime in y]) for y in Y]).unsqueeze(-1)
        else:
            return torch.stack([self.score(y) for y in Y])

class LangermannFunction(TestFunction):

    def __init__(self, input_dim, output_dim, normalized=False):

        self.normalized = normalized
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lower_bounds = [0.0]*input_dim
        self.upper_bounds = [10.0]*input_dim

        self.A = torch.tensor([[3, 5, 2, 1, 7], [5, 2, 1, 4, 9]]) # Original A matrix for Langermann function
        self.A = self.A.repeat(input_dim // 2, output_dim // 5)  # Need to repeat for high-dimensional modification
        self.c = torch.tensor([1, 2, 5, 2, 3]).float()  # Original c vector for the Langermann function
        self.c = self.c.repeat(output_dim // 5,) # Need to repeat foor high-dimensional modification

    def function(self, x):
        if self.normalized:
            xnew = x.clone()
            for i in range(len(xnew)):
                xnew[i] = (xnew[i] * (self.upper_bounds[i] - self.lower_bounds[i])) + self.lower_bounds[i]
        else:
            xnew = x

        x_repeats = torch.cat([xnew.reshape(self.input_dim, 1)] * self.output_dim, -1)
        fo = ((x_repeats - self.A) ** 2).sum(0)
        return fo

    def evaluate(self, X):
        return torch.stack([self.function(x) for x in X])

    def utility(self, y):
        reward = self.c * torch.exp((-1 * y) / math.pi) * torch.cos(math.pi * y)
        out = reward.sum(dim=-1,keepdim=True)
        return -out.reshape(out.shape[0], out.shape[1], 1)

class EnvModelFunction(TestFunction):

    def __init__(self, input_dim, output_dim, normalized = False):

        self.normalized = normalized

        if input_dim is None:
            self.input_dim = 15
        else:
            self.input_dim = input_dim
        
        if output_dim is None:
            self.output_dim = 16
        else:
            self.output_dim = output_dim

        assert input_dim >= 4, "At least 4 input dims are required"
        if not (output_dim == 12):
            assert (
                output_dim & (output_dim - 1) == 0
            ), "Output dim must either be 12 or a power of 2"

        M0 = torch.tensor(10.0, **tkwargs)
        D0 = torch.tensor(0.07, **tkwargs)
        L0 = torch.tensor(1.505, **tkwargs)
        tau0 = torch.tensor(30.1525, **tkwargs)
        if output_dim == 12:
            self.s_size = 3
            self.t_size = 4
        else:
            # Otherwise s and t sizes are root output dim
            self.s_size = int(output_dim**0.5)
            self.t_size = int(output_dim**0.5)
            # Make sure output dim is indeed a power of 2
            assert output_dim == self.s_size * self.t_size
        if self.s_size == 3:
            S = torch.tensor([0.0, 1.0, 2.5], **tkwargs)
        else:
            S = torch.linspace(0.0, 2.5, self.s_size, **tkwargs)
        if self.t_size == 4:
            T = torch.tensor([15.0, 30.0, 45.0, 60.0], **tkwargs)
        else:
            T = torch.linspace(15.0, 60.0, self.t_size, **tkwargs)

        self.Sgrid, self.Tgrid = torch.meshgrid(S, T)
        self.c_true = self.env_cfun(self.Sgrid, self.Tgrid, M0, D0, L0, tau0)
        # Bounds used to unnormalize x (optimize in 0 to 1 range for all)
        self.lower_bounds = [7.0, 0.02, 0.01, 30.010]
        self.upper_bounds = [13.0, 0.12, 3.00, 30.295]

    def env_cfun(self, s, t, M, D, L, tau):
        c1 = M / torch.sqrt(4 * math.pi * D * t)
        exp1 = torch.exp(-(s**2) / 4 / D / t)
        term1 = c1 * exp1
        c2 = M / torch.sqrt(4 * math.pi * D * (t - tau))
        exp2 = torch.exp(-((s - L) ** 2) / 4 / D / (t - tau))
        term2 = c2 * exp2
        term2[torch.isnan(term2)] = 0.0
        return term1 + term2
    
    def function(self, x):

        if self.normalized:
            xnew = x[0:4].clone()
            for i in range(4):
                xnew[i] = (
                    xnew[i] * (self.upper_bounds[i] - self.lower_bounds[i])
                ) + self.lower_bounds[i]
        else:
            xnew = x
            
        return self.env_cfun(self.Sgrid, self.Tgrid, *xnew)
    
    def evaluate(self, X):
        return torch.stack([self.function(x) for x in X])

    def utility(self, y):

        # Resizing the inputs
        if y.shape[-1] == (self.s_size * self.t_size):
            y = y.unsqueeze(-1).reshape(
                *y.shape[:-1], self.s_size, self.t_size
            )

        # Evaluating the utility
        sq_diffs = (y - self.c_true).pow(2)
        return sq_diffs.sum(dim=(-1, -2)).mul(-1.0)

    "Method to plot the contours of the function along with the target contours"
    def optresult_plotter(self, x_list, color_list, label_list, linestyle_list, plot_target = True):

        fig, ax = plt.subplots(dpi=2**8)
        h_list = []
        for i in range(len(x_list)):
            c = ax.contour(self.Sgrid.detach().cpu().numpy(), self.Tgrid.detach().cpu().numpy(), self.function(x_list[i]).detach().cpu().numpy(), 
                            colors = color_list[i], linestyles = linestyle_list[i], linewidth = 0.75, levels = 15)
            h, _ = c.legend_elements()
            h_list.append(h[0])
        
        if plot_target:
            c_target = ax.contour(self.Sgrid.detach().cpu().numpy(), self.Tgrid.detach().cpu().numpy(), self.c_true.detach().cpu().numpy(), colors = 'k', linestyles = 'dashed', 
            levels = 15)
            h_target, _ = c_target.legend_elements()
            h_list.append(h_target[0])
        label_list.append('Target')
        ax.legend(h_list, label_list, ncol = 2, fancybox=True)
        ax.set_xlabel('s')
        ax.set_ylabel('t')
        plt.tight_layout()
        plt.savefig('prediction.pdf')
        plt.show()

    "Method to plot predicted and true contours given a list of models"
    def prediction_plotter(self, x, model_list, color_list, label_list, linestyle_list, save_filename, plot_true = True):

        fig, ax = plt.subplots(dpi=2**8)
        h_list = []
        for i in range(len(model_list)):
            c = ax.contour(self.Sgrid.detach().cpu().numpy(), self.Tgrid.detach().cpu().numpy(), model_list[i].predictROM(x)[0].reshape((self.s_size, self.t_size)).detach().cpu().numpy(), 
                            colors = color_list[i], linestyles = linestyle_list[i], levels = 15)
            h, _ = c.legend_elements()
            h_list.append(h[0])

        if plot_true:
            c_exact = ax.contour(self.Sgrid.detach().cpu().numpy(), self.Tgrid.detach().cpu().numpy(), self.function(x[0]).detach().cpu().numpy(), colors = 'k', linestyles = 'dashed', levels = 15)
            h_exact, _ = c_exact.legend_elements()
            h_list.append(h_exact[0])
        label_list.append('Exact')
        plt.legend(h_list, label_list, ncol = 2)
        ax.set_xlabel('s')
        ax.set_ylabel('t')
        plt.savefig(save_filename)
        plt.show()

class BrusselatorPDE(TestFunction):

    def __init__(self, Nx, Ny, input_dim, normalized = True):

        self.normalized = normalized
        # Setting the paramters for the grid of the PDE
        self.Nx = Nx
        self.Ny = Ny

        self.lower_bounds = [0.1, 0.1, 0.1, 0.01]
        self.upper_bounds = [5.0, 5.0, 5.0, 5.0]

        self.input_dim = input_dim
        self.output_dim = 2*self.Nx*self.Ny

    def function(self, x):

        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()

        if self.normalized:
            xnew = x[0:4].copy()
            for i in range(4):
                xnew[i] = (
                    xnew[i] * (self.upper_bounds[i] - self.lower_bounds[i])
                ) + self.lower_bounds[i]
        else:
            xnew = x

        a = xnew[0]
        b = xnew[1]
        d0 = xnew[2]
        d1 = xnew[3]

        eq = PDE(
            {
                "u": f"{d0} * laplace(u) + {a} - ({b} + 1) * u + u**2 * v",
                "v": f"{d1} * laplace(v) + {b} * u - u**2 * v",
            }
        )

        # initialize state
        grid = UnitGrid([self.Nx, self.Ny])
        u = ScalarField(grid, a, label="Field $u$")
        v = b / a + 0.1 * ScalarField.random_normal(grid, label="Field $v$")
        state = FieldCollection([u, v])

        sol = eq.solve(state, t_range=20, dt=1e-3)
        
        sol_tensor = []
        sol_tensor.append(sol[0].data)
        sol_tensor.append(sol[1].data)
        sol_tensor = np.array(sol_tensor)
        
        ss = sol_tensor[np.isnan(sol_tensor)]
        sol_tensor[np.isnan(sol_tensor)] = 1e5 * np.random.randn(*ss.shape)
        
        return torch.tensor(sol_tensor, **tkwargs)
    
    def evaluate(self, X):
        return torch.stack([self.function(x) for x in X])

    def score(self, y):
        weighting = torch.ones((2,self.Nx,self.Ny), **tkwargs)/10
        weighting[:, [0, 1, -2, -1], :] = 1.0
        weighting[:, :, [0, 1, -2, -1]] = 1.0
        weighted_samples = weighting * y
        return -weighted_samples.var(dim=(-1, -2, -3))

    def utility(self, Y):
        
        # Resizing the inputs
        if Y.shape[-1] == (self.output_dim):
            Y = Y.unsqueeze(-1).unsqueeze(-1).reshape(
                *Y.shape[:-1], 2, self.Nx, self.Ny
            )

        if Y.dim() == 3:
            Y = Y.unsqueeze(0).reshape(1, 2, self.Nx, self.Ny)
        
        return torch.stack([self.score(y) for y in Y])

    "Method to plot predicted and true contours given a list of models"
    def prediction_plotter(self, x, model_list, color_list, label_list, index = 1, plot_true = True):

        X = np.linspace(0,63,64)
        Y = np.linspace(0,63,64)
        xplot, yplot = np.meshgrid(X, Y, indexing='ij')

        fig, ax = plt.subplots(dpi=2**8)
        h_list = []
        for i in range(len(model_list)):
            prediction = model_list[i].predictROM(x)[0].reshape((2, self.Nx, self.Ny)).detach().cpu().numpy()
            c = ax.contour(xplot, yplot, prediction[index,:,:], levels = 15, colors=color_list[i])
            h, _ = c.legend_elements()
            h_list.append(h[0])

        if plot_true:
            exact = self.function(x[0]).detach().cpu().numpy()
            c_exact = ax.contour(xplot, yplot, exact[index,:,:], colors = 'k', levels = 15)
            h_exact, _ = c_exact.legend_elements()
            h_list.append(h_exact[0])

        label_list.append('Exact')
        plt.legend(h_list, label_list)
        ax.set_xlabel('s')
        ax.set_ylabel('t')
        plt.show()

class InverseAirfoil(TestFunction):

    "Class definition for inverse airfoil design test problem - assumes that data is generated using blackbox"

    def __init__(self, directory, airfoil, upper_bounds, lower_bounds, targetCp, normalized = False):

        # Load data from the directory while initiliazing the class
        fieldfile = os.path.join(directory, 'fieldData.mat')
        field_data = loadmat(fieldfile)
        self.xdoe = field_data['x']
        if normalized:
            self.xdoe = (self.xdoe - lower_bounds)/(upper_bounds - lower_bounds)
        self.coefpressure = field_data['CoefPressure']
        self.targetCp = targetCp
        self.normalized = normalized

        # Calculating objective values for initial doe
        self.ydoe = self.utility(torch.tensor(self.coefpressure, **tkwargs))

        # Setting blackbox airfoil class
        self.airfoil = airfoil

        # Setting the upper and lower bounds
        self.lowerbounds = lower_bounds
        self.upperbounds = upper_bounds

    def function(self, x):

        if self.normalized:
            xnew = x.clone()
            for i in range(xnew.shape[-1]):
                xnew[i] = (
                    xnew[i] * (self.upperbounds[i] - self.lowerbounds[i])
                ) + self.lowerbounds[i]
        else:
            xnew = x

        xnew = xnew.detach().cpu().numpy()
        output, fieldData = self.airfoil.getObjectives(xnew)
            
        return torch.tensor(fieldData['CoefPressure'], **tkwargs).unsqueeze(0)
    
    def evaluate(self, x):
        pass

    "Method to calculate the L2-norm inverse design objective"
    def utility(self, y):
        
        # Resize input samples from posterior
        out = torch.square(torch.linalg.norm(y - self.targetCp, ord = 2, dim = -1))

        return -out
