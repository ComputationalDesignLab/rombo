"""
    Definition of base calss for ROM architectures

"""

from abc import ABC, abstractmethod
import torch

# Setting data type and device for Pytorch based libraries
tkwargs = {
    "dtype": torch.float,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

class ROM(ABC):

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
    
    "Method to check whether a given variable is a Tensor or not"
    def _checkTensor(self, x):
        
        if not torch.is_tensor(x):
            x = torch.tensor(x, **tkwargs)
        
        return x

    "Method to fit the ROM to the given data"
    @abstractmethod
    def trainROM(self):
        pass

    "Method to predict using the trained ROM for a given test data"
    @abstractmethod
    def predictROM(self):
        pass
