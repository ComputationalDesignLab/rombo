import torch
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.optim import optimize_acqf
from .basebo import BaseBO

# Setting data type and device for Pytorch based libraries
tkwargs = {
    "dtype": torch.float64,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}
    
class ROMBO(BaseBO):

    "Class definition for ROMBO - utilizes BoTorch to do the calculations and maximization of the acquisition function"

    def __init__(self, init_x, init_y, num_samples, MCObjective, bounds, acquisition, ROM, ROM_ARGS):

        self.xdoe = self._checkTensor(init_x)
        self.ydoe = self._checkTensor(init_y)
        self.bounds = bounds
        self.num_samples = num_samples
        self.acquisition = acquisition
        self.MCObjective = MCObjective
        self.rom = ROM
        self.args = ROM_ARGS

    "Method to set the objective for MC Bayesian optimization"
    def setobjective(self, model):

        if self.args["standard"] == True:
            "Function definition for MC Objective"
            def function(samples, X=None):

                samples = model.dimensionreduction.backmapping(samples)
                samples = model.unstandardize(samples)
                return self.MCObjective.utility(samples)
        
        else:
            "Function definition for MC Objective"
            def function(samples, X=None):

                samples = model.dimensionreduction.backmapping(samples)
                return self.MCObjective.utility(samples)

        self.objective = GenericMCObjective(function)

    "Method to perform only one iteration for running parallel cases with multiple optimizers and multiple settings"
    def do_one_step(self, tag, tkwargs):

        self.best_f = self.MCObjective.utility(self.ydoe).max().item()
        self.best_x = self.MCObjective.utility(self.ydoe).argmax().item()
        print("\nBest Objective Value for {}:".format(tag), self.best_f)
        print("Best Design for {}:".format(tag), self.xdoe[self.best_x])

        self.rom_model = self.rom(self.xdoe, self.ydoe, **self.args)

        # Training the ROM
        self.rom_model.trainROM(verbose=False)
        self.setobjective(self.rom_model)

        # Creating the acquisition function
        sampler = SobolQMCNormalSampler(sample_shape = torch.Size([self.num_samples]))
        acqf = self.setacquisition(model = self.rom_model.gp_model.model, sampler=sampler, best_f=self.best_f)

        # Optimizing the acquisition function to obtain a new point
        new_x, self.maxEI = self.optimize_acquistion_torch(acqf, self.bounds, tkwargs)

        # Storing prediction of ROM for the field and the utility function of the problem
        self.current_prediction = self.rom_model.predictROM(new_x)
        self.utility_prediction = self.MCObjective.utility(self.current_prediction)

        if self.args['saas'] == True:
            self.lengthscales = self.rom_model.gp_model.model.median_lengthscale.detach().cpu().numpy()

        # Add in new data to the existing dataset 
        for x in new_x:
            self.xdoe = torch.cat((self.xdoe, x.unsqueeze(0)), dim = 0)
            new_y = self.MCObjective.function(x)
            new_y = new_y.reshape((1, self.ydoe.shape[-1]))
            self.ydoe = torch.cat((self.ydoe, new_y), dim = 0)

    "Method to run the optimization"
    def optimize(self, tag, n_iterations, tkwargs):

        for iteration in range(n_iterations):

            print("\n\n##### Running iteration {} out of {} #####".format(iteration+1, n_iterations))
            self.do_one_step(tag, tkwargs)


        





        


