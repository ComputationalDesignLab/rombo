import torch
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.optim import optimize_acqf
from .basebo import BaseBO

# Setting data type and device for Pytorch based libraries
tkwargs = {
    "dtype": torch.float,
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

        rom_model = self.rom(self.xdoe, self.ydoe, **self.args)

        # Training the ROM
        rom_model.trainROM(verbose=False)
        self.setobjective(rom_model)

        # Creating the acquisition function
        sampler = SobolQMCNormalSampler(sample_shape = torch.Size([self.num_samples]))
        acqf = self.setacquisition(model = rom_model.gp_model.model, sampler=sampler, best_f=self.best_f)

        # Optimizing the acquisition function to obtain a new point
        new_x, self.maxEI = self.optimize_acquistion_torch(acqf, self.bounds, tkwargs)

        # Storing prediction of ROM for the field and the utility function of the problem
        self.current_prediction = rom_model.predictROM(new_x)
        self.utility_prediction = self.MCObjective.utility(self.current_prediction)

        if self.args['saas'] == True:
            self.lengthscales = rom_model.gp_model.model.median_lengthscale.detach().cpu().numpy()

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

class ROMBO_BAxUS(BaseBO):

    "Class definition for ROMBO with BAxUS embedding - utilizes BoTorch to do the calculations and maximization of the acquisition function"

    def __init__(self, init_x, init_y, num_samples, MCObjective, bounds, acquisition, ROM, ROM_ARGS, state, embedding):

        # Specifying the data for baxus method
        self.xdoe = self._checkTensor(init_x)
        self.ydoe = self._checkTensor(init_y)
        self.x_baxus_input = self.xdoe @ embedding
        self.y_baxus = self.MCObjective.evaluate(self.x_baxus_input)

        self.state = state
        self.embedding = embedding
        self.bounds = bounds
        self.num_samples = num_samples
        self.acquisition = acquisition
        self.MCObjective = MCObjective
        self.rom = ROM
        self.args = ROM_ARGS

    "Method to set the objective for MC Bayesian optimization"
    def setobjective(self, model):

        "Function definition for MC Objective"
        def function(samples, X=None):

            samples = model.dimensionreduction.backmapping(samples)
            return self.MCObjective.utility(samples)

        self.objective = GenericMCObjective(function)

    "Method to run one optimization step using the BAxUS embedding"
    def do_one_step_baxus(self, tag):
        
        self.best_f = self.MCObjective.utility(self.ydoe).max().item()
        self.best_x = self.MCObjective.utility(self.ydoe).argmax().item()
        print("\nBest Objective Value for {}:".format(tag), self.best_f)
        print("Best Design for {}:".format(tag), self.xdoe[self.best_x])

        self.rom_model = self.rom(self.xdoe, self.y_baxus, **self.args)

        # Training the ROM
        self.rom_model.trainROM(verbose=False)
        self.setobjective(self.rom_model)

        # Scale the TR to be proportional to the lengthscales
        x_center = self.xdoe[self.best_x, :].clone()
        weights = self.rom_model.gp_model.model.covar_module.lengthscale.detach().view(-1)
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * self.state.length, -1.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * self.state.length, -1.0, 1.0)

        # Creating the acquisition function
        sampler = SobolQMCNormalSampler(sample_shape = torch.Size([self.num_samples]))
        acqf = self.setacquisition(model = self.rom_model.gp_model.model, sampler=sampler, best_f=self.best_f)

        # Optimizing the acquisition function to obtain a new point
        tr_bounds = torch.stack([tr_lb, tr_ub])
        new_x, self.maxEI = self.optimize_acquistion_torch(acqf, tr_bounds, tkwargs)

    "Method to run the optimization"
    def optimize(self, tag, n_iterations, tkwargs):

        for iteration in range(n_iterations):

            print("\n\n##### Running iteration {} out of {} #####".format(iteration+1, n_iterations))
            self.do_one_step(tag, tkwargs)

class ConstrainedROMBO(BaseBO):

    "Class definition for constrained ROMBO currently equipped to handle single constraint calculated from the multiple outputs of the model"
    def __init__(self, init_x, init_y, cons_y, scores, num_samples, MCObjective, lowerBounds, upperBounds, acquisition, ROM, ROM_ARGS):

        self.xdoe = init_x
        self.ydoe = init_y
        self.ycons = cons_y
        self.scores = scores
        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds
        self.num_samples = num_samples
        self.acquisition = acquisition
        self.MCObjective = MCObjective
        self.rom = ROM
        self.args = ROM_ARGS

    "Method to clamp current doe to the feasible set"
    def clamp_to_feasible(self):

        ydoe_prime = self.scores.clone()
        xdoe_prime = self.xdoe.clone()

        idx = (self.ycons <= 0)

        return ydoe_prime[idx], xdoe_prime[idx.reshape(1,-1)[0], :]

    "Method defining the objective function for pymoo"
    def objective_func(self, x):

        # Reshaping DVs and converting to tensor
        x_reshape = x.reshape((1, x.shape[0]))
        x_tensor = torch.tensor(x_reshape, **tkwargs)

        # Evaluating acquisition function value
        acqf_value = self.acqf(x_tensor)

        return -acqf_value.item()

    "Method to generate a function for each of the constraints"
    def _generate_constraint(self):

        def cons_fun(x):

            # Reshaping and converting to tensor
            x_reshape = x.reshape((1, x.shape[0]))
            x_tensor = torch.tensor(x_reshape, **tkwargs)

            # Predicting outputs
            outputs = self.rom_model.predictROM(x_tensor) 
            if self.args['saas'] == True:
                outputs = outputs.mean(0)

            # Calculating constraint scores
            scores = self.MCObjective.constraint_utility(outputs)

            return scores.detach().cpu().numpy()

        return cons_fun

    "Method to set the objective for MC Bayesian optimization"
    def setobjective(self, model):

        "Function definition for MC Objective"
        def function(samples, X=None):

            if self.args['saas'] == True:
                samples = samples.mean(2)
            
            samples = samples.reshape((self.num_samples, 10))
            samples = model.dimensionreduction.backmapping(samples)

            return self.MCObjective.utility(samples).reshape((self.num_samples,1,1))

        self.objective = GenericMCObjective(function)

    "Method to perform only one iteration for running parallel cases with multiple optimizers and multiple settings"
    def do_one_step(self, tag):

        f_clamped, x_clamped = self.clamp_to_feasible()
        self.best_f = f_clamped.max().item()
        self.best_x = f_clamped.argmax().item()
        print("\nBest Objective Value for {}:".format(tag), self.best_f)
        print("Best Design for {}:".format(tag), x_clamped[self.best_x])

        self.rom_model = self.rom(self.xdoe, self.ydoe, **self.args)

        # Training the ROM
        print("##### Training the ROM Model")
        self.rom_model.trainROM(verbose=False)
        self.setobjective(self.rom_model)

        # Creating the acquisition function and initial conditions
        sampler = SobolQMCNormalSampler(sample_shape = torch.Size([self.num_samples]))
        self.acqf = self.setacquisition(model = self.rom_model.gp_model.model, sampler=sampler, best_f=self.best_f)

        cons_func_list = [self._generate_constraint()]

        # Optimizing the acquisition function to obtain a new point
        print("##### Locating new infill point")
        new_x, _ = self.optimize_acquistion_pymoo(self.objective_func, num_dv=self.MCObjective.input_dim, lowerBounds=self.lowerBounds, upperBounds=self.upperBounds,
                                                  cons_func_list=cons_func_list)

        # Add in new data to the existing dataset 
        new_x =  torch.tensor(new_x, **tkwargs)
        self.xdoe = torch.cat((self.xdoe, new_x.unsqueeze(0)), dim = 0)
        new_y = self.MCObjective.function(new_x)
        new_cons = self.MCObjective.constraint_utility(new_y)
        new_y = new_y.reshape((1, self.ydoe.shape[-1]))
        self.ydoe = torch.cat((self.ydoe, new_y), dim = 0)
        self.ycons = torch.cat((self.ycons, new_cons), dim = 0)
        self.scores = self.MCObjective.utility(self.ydoe)
            
    "Method to run the optimization"
    def optimize(self, tag, n_iterations, tkwargs):

        for iteration in range(n_iterations):

            print("\n\n##### Running iteration {} out of {} #####".format(iteration+1, n_iterations))
            self.do_one_step(tag, tkwargs)


        


