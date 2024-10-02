from baseproblem import TestFunction
import torch
import math

class EnvModelFunction(TestFunction):

    def __init__(self, input_dim, output_dim):

        "Constructor for the class inspired by JoCo and BoTorch codes"

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

        return self.env_cfun(self.Sgrid, self.Tgrid, *x)
    
    def evaluate(self, X):

        return torch.stack([self.env_cfun(self.Sgrid, self.Tgrid, *x) for x in X])

    def utility(self, y):

        # Resizing the inputs
        if y.shape[-1] == (self.s_size * self.t_size):
            y = y.unsqueeze(-1).reshape(
                *y.shape[:-1], self.s_size, self.t_size
            )

        # Evaluating the utility
        sq_diffs = (y - self.c_true).pow(2)
        return sq_diffs.sum(dim=(-1, -2))
    
class BrusselatorPDE(TestFunction):

    pass



