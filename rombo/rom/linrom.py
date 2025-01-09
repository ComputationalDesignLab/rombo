"""
Creation of a ROM model using parameters given by a user.

Current Implementation:
    
    - Linear nonintrusive ROM using POD/PCA

"""
import torch
from ..interpolation.interpolation import BoTorchModelList
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

        # Training GPR model
        self.gp_model = BoTorchModelList(self.low_dim_model, self.low_dim_likelihood, train_x, a.T.detach(), model_args={"outcome_transform": Standardize(m=1)})
        self.gp_model.train()

    "Method to predict using the trained ROM for a given test data"
    def predictROM(self, test_x):

        # Prediction for the low dimensional space
        predicted_a, variances = self.gp_model.predict(test_x)

        # Backmapping and unstandardizing
        field = self.dimensionreduction.backmapping(predicted_a)
        return field