"""
Creation of a ROM model using parameters given by a user.

Current Implementation:
    
    - Linear nonintrusive ROM using POD/PCA

"""
import torch
from ..interpolation.interpolation import GPRModel
from ..dimensionality_reduction.dim_red import LinearReduction
from .baserom import ROM

# Setting data type and device for Pytorch based libraries
tkwargs = {
    "dtype": torch.float,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

class PODROM(ROM):
    
    def __init__(self, param_doe, snapshot_matrix, ric, low_dim_model, low_dim_likelihood, standard = True):
        
        # Setting the training data of the model
        self.param_doe = self._checkTensor(param_doe)
        self.high_dim_data = self._checkTensor(snapshot_matrix)
        self.standard = standard

        # Setting the interpolation and regression data for the model
        self.low_dim_model = low_dim_model
        self.low_dim_likelihood = low_dim_likelihood

        # Setting the dimensionality reduction method and corresponding data
        if standard:
            self.train_y = self.standardize(self.high_dim_data)
        else:
            self.train_y = self.high_dim_data
        self.dimensionreduction = LinearReduction(self.train_y, ric)
    
    "Method to fit the ROM to the given data"
    def trainROM(self, verbose):

        # Setting training data
        train_x = self.param_doe

        # Fitting the dimensionality reduction
        self.phi, k = self.dimensionreduction.fit()

        # Computing the encoding for the data
        a = self.dimensionreduction.encoding(self.phi)

        # Training GPR model
        self.gp_model = GPRModel(self.low_dim_model, self.low_dim_likelihood, train_x, a.T, tkwargs)
        self.gp_model.train(num_epochs=500, verbose=verbose)

    "Method to predict using the trained ROM for a given test data"
    def predictROM(self, test_x):

        # Prediction for the low dimensional space
        predicted_a, lower, upper = self.gp_model.predict(test_x)

        # Backmapping and unstandardizing
        field = self.dimensionreduction.backmapping(self.phi, predicted_a)
        if self.standard:
            field = self.unstandardize(field.T)
            return field
        else:
            return field
        

        
        
        
