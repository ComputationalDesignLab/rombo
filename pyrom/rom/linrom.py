"""
Creation of a ROM model using parameters given by a user.

Current Implementation:
    
    - Linear nonintrusive ROM using POD/PCA

"""
import torch
import gpytorch
from interpolation.models import MultitaskGPModel, ExactGPModel
from interpolation.interpolation import GPyTorchModel

# Setting data type and device for Pytorch based libraries
tkwargs = {
    "dtype": torch.float,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

class PODROM():
    
    def __init__(self, params):
        
        # Initializing parameters of the model
        self.params = params

    def _settrainingdata(self, param_doe, high_dim_data):

        "Method to set training data of the model"
        self.param_doe = self._checkTensor(param_doe)
        self.high_dim_data = self._checkTensor(high_dim_data)

    def computeRIC(self, s_full):

        "Method to select the best number of POD modes depending on relative information content"
        s_full_sq = torch.square(s_full)
        
        for k in range(len(s_full)):
            
            s_trunc = s_full[0:k]
            s_trunc_sq = torch.square(s_trunc)
            
            RIC = torch.sum(s_trunc_sq)/torch.sum(s_full_sq)
            
            if RIC > self.params['RIC_delta']:
                break
            else:
                continue
            
        return k

    def computePOD(self):

        "Method to compute the singular value decomposition of the provided snapshot matrix"
        phi, s, psi = torch.linalg.svd(self.high_dim_data, True)
        k = self.computeRIC(s)
        phi_trunc = phi[:,0:k]
        
        return phi_trunc, k
    
    def computeCOEFF(self, phi_k):

        "Method to compute coefficients for truncated POD modes"
        Wr = torch.matmul(phi_k.T, W) 
        
        return Wr
    
    def _backmap(self, phi_trunc, a):

        "Method to perform a linear backmapping to high-dimensional space"
        field = torch.matmul(phi_trunc, a.T)

        return field
    
    def standardize(self, Y):

        "Method to perform standardization of data"
    
        stddim = -1 if Y.dim() < 2 else -2
        Y_std = Y.std(dim=stddim, keepdim=True)
        Y_mean = Y.mean(dim=stddim, keepdim=True)
        Y_standard = (Y - Y_mean) / Y_std
        
        return Y_standard, Y_mean, Y_std
        
    def unstandardize(self, Y, Y_mean, Y_std):

        "Method to unstandardize the data"

        return Y*Y_std + Y_mean
    
    def trainROM(self):

        "Method to fit the ROM to the given data"

        # Standardization of snapshot matrix
        if self.params["standardize"]:
            train_y, self.y_mean, self.y_std = self.standardize(self.high_dim_data)
        else:
            train_y = self.high_dim_data

        # Setting training data
        train_x = self.param_doe

        # Performing dimensionality reduction
        phi_trunc, self.k = self.computePOD(train_y.T)
        W_r = self.computeCOEFF(phi_trunc, train_y.T)
        W_r_std, self.w_mean, self.w_std = self.standardize(W_r.T)

        # Training GPR model
        low_dim_model = GPyTorchModel() 





            
        

    def _checkTensor(self, x):
        
        "Method to check whether a given variable is a Tensor or not"
        
        if not torch.is_tensor(x):
            x = torch.tensor(x, **self.tkwargs)
        
        return x
        

        
        
        
