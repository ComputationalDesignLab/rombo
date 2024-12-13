"""
Creation of a ROM model using parameters given by a user.

Current Implementation:
    
    - Linear nonintrusive ROM using POD/PCA

"""
import torch
import gpytorch
from ..interpolation.interpolation import GPRModel, BoTorchModel
from ..dimensionality_reduction.dim_red import LinearReduction
from .baserom import ROM
from botorch.models.transforms import Standardize

# Setting data type and device for Pytorch based libraries
tkwargs = {
    "dtype": torch.float64,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

class PODROM(ROM):
    
    def __init__(self, param_doe, snapshot_matrix, ric, low_dim_model, low_dim_likelihood, saas = False):
        
        # Setting the training data of the model
        self.param_doe = self._checkTensor(param_doe)
        self.high_dim_data = self._checkTensor(snapshot_matrix)
        self.saas = saas

        # Setting the interpolation and regression data for the model
        self.low_dim_model = low_dim_model
        self.low_dim_likelihood = low_dim_likelihood

        # Setting the dimensionality reduction method and corresponding data
        self.train_y = self.standardize(self.high_dim_data)
        self.dimensionreduction = LinearReduction(self.train_y, ric, self.Y_mean, self.Y_std)
    
    "Method to fit the ROM to the given data"
    def trainROM(self, verbose):

        # Setting training data
        train_x = self.param_doe

        # Fitting the dimensionality reduction
        self.phi, self.k = self.dimensionreduction.fit()

        # Computing the encoding for the data
        a = self.dimensionreduction.encoding(self.phi)
        a = a.T
        a_prime = a.repeat(a.shape[-1],1,1).reshape(a.shape[-1], train_x.shape[0]*a.shape[-1], 1)
        train_x = train_x.repeat(a.shape[-1],1,1)
        print(train_x.shape)
        print(a_prime.shape)

        if self.saas:
            I = []
            for i in range(a.T.shape[-1]):
                I.append(i*torch.ones((len(train_x), 1), **tkwargs))
            train_X = torch.cat([torch.cat([train_x, i], -1) for i in I])
            train_Y = a.detach().flatten(0).reshape((len(train_X), 1))

            # Training GPR model
            self.gp_model = BoTorchModel(self.low_dim_model, self.low_dim_likelihood, train_X, train_Y, model_args={"task_feature": -1, "outcome_transform": Standardize(train_Y.shape[-1])})
            self.gp_model.train(type='bayesian')
        else:
            # Training GPR model
            self.gp_model = BoTorchModel(self.low_dim_model, self.low_dim_likelihood, train_x, a_prime.detach(), model_args={"likelihood":gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([9]))})#"outcome_transform": Standardize(a_prime.detach().shape[-1])})
            self.gp_model.train(type='mll')

    "Method to predict using the trained ROM for a given test data"
    def predictROM(self, test_x):

        # Prediction for the low dimensional space
        predicted_a, variances = self.gp_model.predict(test_x)

        # Backmapping and unstandardizing
        field = self.dimensionreduction.backmapping(predicted_a)
        return field
        

        
        
        