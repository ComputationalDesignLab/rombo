from .baseproblem import TestFunction
import os
import torch
import numpy as np
from scipy.io import loadmat
import math
import matplotlib.pyplot as plt 
from pde import PDE, FieldCollection, ScalarField, UnitGrid

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

        M0 = torch.tensor(10.0).float()
        D0 = torch.tensor(0.07).float()
        L0 = torch.tensor(1.505).float()
        tau0 = torch.tensor(30.1525).float()
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
            S = torch.tensor([0.0, 1.0, 2.5]).float()
        else:
            S = torch.linspace(0.0, 2.5, self.s_size).float()
        if self.t_size == 4:
            T = torch.tensor([15.0, 30.0, 45.0, 60.0]).float()
        else:
            T = torch.linspace(15.0, 60.0, self.t_size).float()

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
    def optresult_plotter(self, x_list, color_list, label_list, plot_target = True):

        fig, ax = plt.subplots(dpi=2**8)
        h_list = []
        for i in range(len(x_list)):
            c = ax.contour(self.Sgrid.detach().cpu().numpy(), self.Tgrid.detach().cpu().numpy(), self.function(x_list[i]).detach().cpu().numpy(), colors = color_list[i], levels = 12)
            h, _ = c.legend_elements()
            h_list.append(h[0])
        
        if plot_target:
            c_target = ax.contour(self.Sgrid.detach().cpu().numpy(), self.Tgrid.detach().cpu().numpy(), self.c_true.detach().cpu().numpy(), colors = 'k', linestyles = 'dashed', 
            levels = 12)
            h_target, _ = c_target.legend_elements()
            h_list.append(h_target[0])
        label_list.append('Target')
        ax.legend(h_list, label_list, ncol = 3, fancybox=True)
        ax.set_xlabel('s')
        ax.set_ylabel('t')
        plt.tight_layout()
        #plt.savefig('prediction.pdf')
        plt.show()

    "Method to plot predicted and true contours given a list of models"
    def prediction_plotter(self, x, model_list, color_list, label_list, save_filename, plot_true = True):

        fig, ax = plt.subplots(dpi=2**8)
        h_list = []
        for i in range(len(model_list)):
            c = ax.contour(self.Sgrid.detach().cpu().numpy(), self.Tgrid.detach().cpu().numpy(), model_list[i].predictROM(x)[0].reshape((self.s_size, self.t_size)).detach().cpu().numpy(), colors = color_list[i], levels = 15)
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

    def __init__(self, Nx, Ny, input_dim, tkwargs, normalized = True):

        self.normalized = normalized
        # Setting the paramters for the grid of the PDE
        self.Nx = Nx
        self.Ny = Ny

        self.lower_bounds = [0.1, 0.1, 0.1, 0.01]
        self.upper_bounds = [5.0, 5.0, 5.0, 5.0]

        self.input_dim = input_dim
        self.output_dim = 2*self.Nx*self.Ny

        self.tkwargs = tkwargs

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
        
        return torch.tensor(sol_tensor, **self.tkwargs)
    
    def evaluate(self, X):
        return torch.stack([self.function(x) for x in X])

    def score(self, y):
        weighting = torch.ones((2,self.Nx,self.Ny))/10
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
            c = ax.contourf(xplot, yplot, prediction[index,:,:], levels = 15)
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

class Airfoil(TestFunction):

    "Class definition for airfoil optimization test problem - assumes that data is generated using blackbox"

    def __init__(self, directory, airfoil, airfoil_x, upper_bounds, lower_bounds, targetCL, base_area, base_thickness, baseline_upper, baseline_lower,
                 tkwargs, normalized = False):

        # Load data from the directory while initiliazing the class
        fieldfile = os.path.join(directory, 'fieldData.mat')
        coefffile = os.path.join(directory, 'data.mat')
        field_data = loadmat(fieldfile)
        coeff_data = loadmat(coefffile)
        self.xdoe = field_data['x']
        self.coefpressure = field_data['CoefPressure']
        self.coefdrag = coeff_data['cd']
        self.coeflift = coeff_data['cl']
        self.area = coeff_data['area']
        self.tkwargs = tkwargs
        self.baseline_upper = baseline_upper
        self.baseline_lower = baseline_lower
        self.base_thickness = base_thickness
        self.targetCL = targetCL
        self.base_area = base_area
        self.normalized = normalized

        # Setting the ydoe to be concatenated pressure and total drag
        self.ydoe = np.hstack([self.coefdrag, self.coefpressure])

        # Setting the x coordinates for the airfoil
        self.airfoil_x = airfoil_x

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
        output, fieldData = self.airfoil.getObjectives(x)
            
        return output, fieldData
    
    "Method for computing y/c given x/c and shape parameters"
    def compute_yc(self, x):

        # Extracting CST coeffs from DV vector
        x_upper = x[1:7]
        x_lower = x[7:]

        # Splitting x coordinate values
        x_c_u = self.airfoil_x[0:250]
        x_c_l = self.airfoil_x[250:500]

        airfoil_upper = self.airfoil.DVGeo.computeCSTCoordinates(x_c_u, 0.5, 1.0, x_upper, 0.0)
        airfoil_lower = self.airfoil.DVGeo.computeCSTCoordinates(x_c_l, 0.5, 1.0, x_lower, 0.0)
        
        airfoil_upper = np.nan_to_num(airfoil_upper)
        airfoil_lower = np.nan_to_num(airfoil_lower)

        y_c = np.concatenate((airfoil_upper, airfoil_lower))

        return y_c
    
    "Method to calculate the area constraint for the airfoil problem"
    def area_constraint(self, x):

        area = self.airfoil.calculateArea(x)
        return 1 - (area/self.base_area)

    "Method to caclulate the thickness constraint for the airfoil problem"
    def thickness_constraint(self, x):

        u_airfoil = self.airfoil.DVGeo.computeCSTCoordinates(np.array([self.airfoil_x]), 0.5, 1.0, self.baseline_upper, 0.0)
        l_airfoil = self.airfoil.DVGeo.computeCSTCoordinates(np.array([self.airfoil_x]), 0.5, 1.0, self.baseline_lower, 0.0)

        t_airfoil = self.base_thickness*(u_airfoil - l_airfoil)

        x_upper = x.detach().cpu().numpy()[1:7]
        x_lower = x.detach().cpu().numpy()[7:]

        u_x = self.airfoil.DVGeo.computeCSTCoordinates(np.array([self.airfoil_x]), 0.5, 1.0, x_upper, 0.0)
        l_x = self.airfoil.DVGeo.computeCSTCoordinates(np.array([self.airfoil_x]), 0.5, 1.0, x_lower, 0.0)

        t_x = u_x - l_x

        return 1 - t_x/t_airfoil
    
    def evaluate(self,x):
        pass

    "Method for integrating the pressure distribution to obtain the lift"
    def integration(self, y):
        
        # Defining tensors for calculating gradients
        pressure = y[0:]
        cn = torch.trapezoid(pressure, torch.tensor(self.airfoil_x, **self.tkwargs)) # Normal force component
        return cn

    def utility(self, y):
        
        # Returning the first element of the array since that is the drag output
        return y[:, 0]

    




