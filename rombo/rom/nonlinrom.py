import torch
from ..interpolation.interpolation import BoTorchModel
from ..dimensionality_reduction.dim_red import AutoencoderReduction
from .baserom import ROM
from botorch.models.transforms import Standardize
from botorch.models.kernels import InfiniteWidthBNNKernel

tkwargs = {"device": torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0"), "dtype": torch.float64}

class AUTOENCROM(ROM):

    def __init__(self, param_doe, snapshot_matrix, autoencoder, low_dim_model, low_dim_likelihood, supervised = False, standard = True, saas = False, ibnn = False, ibnn_depth = 3):
            
        # Setting the training data of the model
        self.param_doe = self._checkTensor(param_doe)
        self.high_dim_data = self._checkTensor(snapshot_matrix)
        self.standard = standard
        self.autoencoder = autoencoder
        self.supervised = supervised
        self.saas = saas
        self.ibnn = ibnn
        self.ibnn_depth = ibnn_depth

        # Setting the interpolation and regression data for the model
        self.low_dim_model = low_dim_model
        self.low_dim_likelihood = low_dim_likelihood

        # Setting the dimensionality reduction method and corresponding data
        if standard:
            self.train_y = self.standardize(self.high_dim_data)
        else:
            self.train_y = self.high_dim_data
        self.dimensionreduction = AutoencoderReduction(self.train_y, autoencoder)

    "Method to fit the ROM to the given data"
    def trainROM(self, verbose, type = 'mll'):

        # Setting training data
        train_x = self.param_doe

        # Fitting the dimensionality reduction
        self.dimensionreduction.fit(epochs = 1000, verbose=verbose)

        # Computing the encoding for the data
        a = self.dimensionreduction.encoding()

        if self.supervised:
            a = torch.concatenate((a, self.autoencoder.xlabels), dim = 1)

        if self.saas:
            I = []
            for i in range(a.shape[-1]):
                I.append(i*torch.ones((len(train_x), 1), **tkwargs))
            train_X = torch.cat([torch.cat([train_x, i], -1) for i in I])
            train_Y = a.T.detach().flatten(0).reshape((len(train_X), 1))

            # Training GPR model
            self.gp_model = BoTorchModel(self.low_dim_model, self.low_dim_likelihood, train_X, train_Y, model_args={"task_feature": -1, "outcome_transform": Standardize(train_Y.shape[-1])})
            self.gp_model.trainModel(type='bayesian')
        else:
            # Training GPR model
            if self.ibnn: # Determining whether to use the infinite width BNN kernel function
                self.gp_model = BoTorchModel(self.low_dim_model, self.low_dim_likelihood, train_x, a.detach(), model_args={"data_covar_module": InfiniteWidthBNNKernel(depth=self.ibnn_depth), "outcome_transform": Standardize(a.detach().shape[-1])})
            else:
                self.gp_model = BoTorchModel(self.low_dim_model, self.low_dim_likelihood, train_x, a.detach(), model_args={"outcome_transform": Standardize(a.detach().shape[-1])})
            self.gp_model.trainModel()

    "Method to predict using the trained ROM for a given test data"
    def predictROM(self, test_x):

        # Prediction for the low dimensional space
        predicted_a, variances = self.gp_model.predict(test_x)

        # Backmapping and unstandardizing
        field = self.dimensionreduction.backmapping(predicted_a)
        if self.standard:
            field = self.unstandardize(field)
            return field
        else:
            return field