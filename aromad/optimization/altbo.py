import torch
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.objective import GenericMCObjective
from .basebo import BaseBO
from ..interpolation.interpolation import BoTorchModel

class BO(BaseBO):

    "Class definition for BayesOpt - this can be used to perform Bayesian optimization using single GP models"

    def __init__(self, init_x, init_y, num_samples, MCObjective, bounds, acquisition, GP, MLL, GP_ARGS = {}, training = 'mll'):

        self.xdoe = init_x
        self.ydoe = init_y
        self.bounds = bounds
        self.num_samples = num_samples
        self.acquisition = acquisition
        self.MCObjective = MCObjective
        self.gp = GP
        self.mll = MLL
        self.training = training
        self.gp_args = GP_ARGS

    def do_one_step(self, tag, tkwargs):

        self.best_f = self.ydoe.max()
        self.best_x = self.ydoe.argmax()
        print("\nBest Objective Value for {}:".format(tag), self.best_f.item())
        print("Best Design for {}:".format(tag), self.xdoe[self.best_x.item()])
        
        # Training the GP model
        gp_model = BoTorchModel(self.gp, self.mll, self.xdoe, self.ydoe, model_args=self.gp_args)
        gp_model.train(type=self.training)

        # Creating the acquisition function
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_samples]))
        acqf = self.setacquisition(model=gp_model.model, sampler=sampler, best_f=self.best_f, objective_required = False)

        # Optimizing the acquisition function to obtain a new point
        new_x, _ = self.optimize_acquistion_torch(acqf, self.bounds, tkwargs)

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


