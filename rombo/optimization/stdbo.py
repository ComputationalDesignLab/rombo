import torch
import torch.nn as nn
from botorch.sampling import SobolQMCNormalSampler
from botorch.models.transforms import Standardize
from .basebo import BaseBO
from ..interpolation.interpolation import BoTorchModel, DeepKernelGP

# Setting data type and device for Pytorch based libraries
tkwargs = {
    "dtype": torch.float64,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

class BO(BaseBO):

    "Class definition for standard BO - this can be used to perform Bayesian optimization using single GP models"

    def __init__(self, init_x, init_y, num_samples, MCObjective, bounds, acquisition, GP, MLL, GP_ARGS = {}, training = 'mll'):

        self.xdoe = self._checkTensor(init_x)
        self.ydoe = self._checkTensor(init_y)
        self.bounds = bounds
        self.num_samples = num_samples
        self.acquisition = acquisition
        self.MCObjective = MCObjective
        self.gp = GP
        self.mll = MLL
        self.training = training
        self.gp_args = GP_ARGS

    def do_one_step(self, tag, tkwargs):

        self.best_f = self.ydoe.max().item()
        self.best_x = self.ydoe.argmax().item()
        print("\nBest Objective Value for {}:".format(tag), self.best_f)
        print("Best Design for {}:".format(tag), self.xdoe[self.best_x])
        
        # Training the GP model
        if self.training == 'bayesian':
            self.gp_args = {"outcome_transform": Standardize(self.ydoe.shape[-1])}
        gp_model = BoTorchModel(self.gp, self.mll, self.xdoe, self.ydoe, model_args=self.gp_args)
        gp_model.trainModel(type=self.training)

        # Creating the acquisition function
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_samples]))
        acqf = self.setacquisition(model=gp_model.model, sampler=sampler, best_f=self.best_f, objective_required = False)

        # Optimizing the acquisition function to obtain a new point
        new_x, self.maxEI = self.optimize_acquistion_torch(acqf, self.bounds, tkwargs)

        if self.training == 'bayesian':
            self.lengthscales = gp_model.model.median_lengthscale.detach().cpu().numpy()

        # Add in new data to the existing dataset 
        for x in new_x:
            self.xdoe = torch.cat((self.xdoe, x.unsqueeze(0)), dim = 0)
            new_y = self.MCObjective.function(x)
            new_y = new_y.reshape((1, new_y.shape[-1]))
            new_score = self.MCObjective.utility(new_y)
            self.ydoe = torch.cat((self.ydoe, new_score.reshape((1,self.ydoe.shape[-1]))), dim = 0)

    "Method to run the optimization in a loop"
    def optimize(self, tag, n_iterations, tkwargs):

        for iteration in range(n_iterations):

            print("\n\n##### Running iteration {} out of {} #####".format(iteration+1, n_iterations))
            self.do_one_step(tag, tkwargs)

class DKLBO(BaseBO):

    def __init__(self, init_x, init_y, num_samples, MCObjective, bounds, acquisition, hidden_dims, latent_dim, scaler):

        self.xdoe = self._checkTensor(init_x)
        self.ydoe = self._checkTensor(init_y)
        self.bounds = bounds
        self.num_samples = num_samples
        self.acquisition = acquisition
        self.MCObjective = MCObjective
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.scaler = scaler

    def do_one_step(self, tag, tkwargs):

        self.best_f = self.ydoe.max().item()
        self.best_x = self.ydoe.argmax().item()
        print("\nBest Objective Value for {}:".format(tag), self.best_f)
        print("Best Design for {}:".format(tag), self.xdoe[self.best_x])

        # Standardizing the output data
        Y_tf, _ = self.scaler(self.ydoe)
        
        # Training the GP model
        gp_model = DeepKernelGP(self.xdoe, Y_tf.reshape(-1), self.hidden_dims, zd=self.latent_dim, outcome_transform=self.scaler, activation=nn.ReLU())
        gp_model.trainModel()

        # Creating the acquisition function
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_samples]))
        acqf = self.setacquisition(model=gp_model, sampler=sampler, best_f=self.best_f, objective_required = False)

        # Optimizing the acquisition function to obtain a new point
        new_x, self.maxEI = self.optimize_acquistion_torch(acqf, self.bounds, tkwargs)

        # Add in new data to the existing dataset
        for x in new_x:
            self.xdoe = torch.cat((self.xdoe, x.unsqueeze(0)), dim = 0)
            new_y = self.MCObjective.function(x)
            new_score = self.MCObjective.utility(new_y)
            self.ydoe = torch.cat((self.ydoe, new_score.reshape((1,self.ydoe.shape[-1]))), dim = 0)

    "Method to run the optimization in a loop"
    def optimize(self, tag, n_iterations, tkwargs):

        for iteration in range(n_iterations):

            print("\n\n##### Running iteration {} out of {} #####".format(iteration+1, n_iterations))
            self.do_one_step(tag, tkwargs)


