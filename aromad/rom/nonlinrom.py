import torch
import gpytorch
from ..interpolation.models import MultitaskGPModel, ExactGPModel
from ..interpolation.interpolation import GPRModel
from ..dimensionality_reduction.dim_red import AutoencoderReduction
from abc import ABC, abstractmethod

# Setting data type and device for Pytorch based libraries
tkwargs = {
    "dtype": torch.float,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

class NONLINROM(ABC):

    "Method to fit the ROM to the given data"
    @abstractmethod
    def trainROM(self):
        pass

    "Method to predict using the trained ROM for a given test data"
    @abstractmethod
    def predictROM(self):
        pass

class AUTOENCROM(NONLINROM):

    def __init__(self, param_doe, snapshot_matrix, autoencoder, low_dim_model, low_dim_likelihood, standard = True):
            
        # Setting the training data of the model
        self.param_doe = self._checkTensor(param_doe)
        self.high_dim_data = self._checkTensor(snapshot_matrix)
        self.standard = standard
        self.autoencoder = autoencoder

        # Setting the interpolation and regression data for the model
        self.low_dim_model = low_dim_model
        self.low_dim_likelihood = low_dim_likelihood

        # Setting the dimensionality reduction method and corresponding data
        if standard:
            self.train_y = self.standardize(self.high_dim_data)
        else:
            self.train_y = self.high_dim_data
        self.dimensionreduction = AutoencoderReduction(self.train_y, autoencoder)
        
    "Method to perform standardization of data"
    def standardize(self, Y):
    
        stddim = -1 if Y.dim() < 2 else -2
        self.Y_std = Y.std(dim=stddim, keepdim=True)
        self.Y_mean = Y.mean(dim=stddim, keepdim=True)
        Y_standard = (Y - self.Y_mean) / self.Y_std
        
        return Y_standard
    
    "Method to unstandardize the data"
    def unstandardize(self, Y):

        return Y*self.Y_std + self.Y_mean

    "Method to fit the ROM to the given data"
    def trainROM(self, verbose):

        # Setting training data
        train_x = self.param_doe

        # Fitting the dimensionality reduction
        self.dimensionreduction.fit(epochs = 1000)

        # Computing the encoding for the data
        a = self.dimensionreduction.encoding()

        # Training GPR model
        self.gp_model = GPRModel(self.low_dim_model, self.low_dim_likelihood, train_x, a.detach(), tkwargs)
        self.gp_model.train(num_epochs=500, verbose=verbose)

    "Method to predict using the trained ROM for a given test data"
    def predictROM(self, test_x):

        # Prediction for the low dimensional space
        predicted_a, lower, upper = self.gp_model.predict(test_x)

        # Backmapping and unstandardizing
        field = self.dimensionreduction.backmapping(predicted_a)
        if self.standard:
            field = self.unstandardize(field)
            return field
        else:
            return field

    "Method to check whether a given variable is a Tensor or not"
    def _checkTensor(self, x):
        
        if not torch.is_tensor(x):
            x = torch.tensor(x, **self.tkwargs)
        
        return x


