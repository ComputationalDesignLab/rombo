import numpy as np
import torch.nn as nn
import torch
from smt.sampling_methods import LHS
from rombo.dimensionality_reduction.autoencoder import MLPAutoEnc
from rombo.rom.nonlinrom import AUTOENCROM
from rombo.test_problems.test_problems import RosenbrockFunction
from botorch.models import KroneckerMultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.metrics import mean_squared_error
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig, ChoiceParameterConfig

# Setting data type and device for Pytorch based libraries
tkwargs = {
    "dtype": torch.float64,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

# Creating the optimization class
class optimizeAutoencoder():

    def __init__(self, trainSamples, trainSize=0.7):

        # Initiliaze the ax client
        self.client = Client()

        # Generating the data

        # Defining environment model function 
        self.problem = RosenbrockFunction(input_dim = 10, output_dim = 18, normalized = True)

        # Creating the training data
        n_data = trainSamples
        xlimits = np.array([[0.0, 1.0]]*self.problem.input_dim)
        sampler = LHS(xlimits=xlimits, criterion="ese", random_state=10)
        xtrain = sampler(n_data)
        xtrain = torch.tensor(xtrain, **tkwargs)
        htrain = self.problem.evaluate(xtrain).flatten(1)

        # Splitting the data into training and validation data
        indices = torch.randperm(trainSamples)
        numTrain = int(trainSize * trainSamples)
        trainIndices = indices[:numTrain]
        valIndices = indices[numTrain:]

        self.trainH = htrain[trainIndices]
        self.trainX = xtrain[trainIndices]
        
        self.valH = htrain[valIndices]
        self.valX = xtrain[valIndices]

        # Activation function dictionary
        self.activation = {1: nn.SiLU(), 2: nn.LeakyReLU(), 3: nn.Tanh(), 4: nn.ReLU()}

        # Specifying the parameters for the model
        self.parameters = [
            RangeParameterConfig(
                name='autoencoder_shrinkage_factor', parameter_type='float', bounds=(0.5,1)
            ),
            ChoiceParameterConfig(
                name = 'autoencoder_hidden_layers', parameter_type='int', values = [i for i in range(1,6)],
            ),
            ChoiceParameterConfig(
                name='autoencoder_first_layer', parameter_type='int', values = [2**i for i in range(4,11)],
            ),
            ChoiceParameterConfig(
                name='autoencoder_latent_dim', parameter_type='int', values = [2**i for i in range(1,5)],
            ),
            ChoiceParameterConfig(
                name='activation', parameter_type='int', values = [1,2,3,4],
            ),
        ]

    def trainModel(self, x1, x2, x3, x4, x5):

        """
            Method to calculate the objective function
        """

        # Setting up the hidden dims
        hidden_dims = []
        hidden_dims.append(x3)
        dim = x3
        for i in range(x2):
            dim = int(dim * x1)
            hidden_dims.append(dim)

        # Generating the nonlinear ROM model
        autoencoder = MLPAutoEnc(high_dim=self.problem.output_dim, hidden_dims=hidden_dims, zd = x4, activation = x5).double()
        rom = AUTOENCROM(self.trainX, self.trainH, autoencoder = autoencoder, low_dim_model = KroneckerMultiTaskGP, low_dim_likelihood = ExactMarginalLogLikelihood)

        # Training the ROM and predicting on the test data
        rom.trainROM(verbose=False)
        field = rom.predictROM(self.valX)

        # Calculating the mean NRMSE for surface pressure
        max_values, _ = torch.max(self.valH, dim=0)
        min_values, _ = torch.min(self.valH, dim=0)
        nrmse = torch.sqrt(torch.mean((self.valH - field)**2, dim=0))/(max_values - min_values)
        nrmse = nrmse.detach().cpu().numpy()

        return np.mean(nrmse)

    def singleRunTrials(self, rounds, numPerRound):

        """
            Method to run trials for the Bayesian optimization
        """

        # Defining a objective
        def objective(x1, x2, x3, x4, x5):
            o1 = self.trainModel(x1, x2, x3, x4, x5)
            return o1 

        # Configuring the parameters and the objective function
        self.client.configure_experiment(parameters=self.parameters)
        metric_name = "objective"
        self.client.configure_optimization(objective=f"-{metric_name}")

        for _ in range(rounds):

            trials = self.client.get_next_trials(max_trials=numPerRound)

            for trial_index, parameters in trials.items():
                x1 = parameters["autoencoder_shrinkage_factor"]
                x2 = parameters["autoencoder_hidden_layers"]
                x3 = parameters["autoencoder_first_layer"]
                x4 = parameters["autoencoder_latent_dim"]
                x5 = self.activation[parameters["activation"]]

                result = objective(x1, x2, x3, x4, x5)

                # Set raw_data as a dictionary with metric names as keys and results as values
                raw_data = {metric_name: result}

                # Complete the trial with the result
                self.client.complete_trial(trial_index=trial_index, raw_data=raw_data)
                print(f"Completed trial {trial_index} with {raw_data=}")

    def getBestParams(self):
        best_parameters, _, _, _ = self.client.get_best_parameterization()
        return best_parameters



        


