"""

Definition of classes for dimensionality reduction using linear and nonlinear methods

- Proper Orthogonal Decomposition
- Autoencoders

"""
import torch
from abc import ABC, abstractmethod

# Setting data type and device for Pytorch based libraries
tkwargs = {
    "dtype": torch.float64,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

class DimensionalityReduction(ABC):

    "Method to assign snapshot vectors and center them if required"
    def _setsnapshots(self, S):

        self.snapshots = S.to(**tkwargs)

    "Method to fit the dimensionality reduction method to the snapshot data"
    @abstractmethod
    def fit(self):
        pass

    "Method to compute encoding of snapshot matrix"
    @abstractmethod
    def encoding(self):
        pass
    
    "Method to reconstruct high dimensional solution from low dimensional representation"
    @abstractmethod
    def backmapping(self):
        pass

class LinearReduction(DimensionalityReduction):

    def __init__(self, S, RIC, mean, std):

        # Relative information content is the only parameter required
        self.ric = RIC
        self._setsnapshots(S)
        self.mean = mean
        self.std = std

    "Method to select the best number of POD modes depending on relative information content"
    def computeRIC(self, s_full):

        s_full_sq = torch.square(s_full)
        
        for k in range(len(s_full)):
            
            s_trunc = s_full[0:k]
            s_trunc_sq = torch.square(s_trunc)
            
            RIC = torch.sum(s_trunc_sq)/torch.sum(s_full_sq)
            
            if RIC > self.ric:
                break
            else:
                continue
            
        return k

    "Method to compute the singular value decomposition of the provided snapshot matrix"
    def fit(self):

        phi, s, psi = torch.linalg.svd(self.snapshots.T, True)
        k = self.computeRIC(s)
        self.phi_trunc = phi[:,0:k]
        
        return self.phi_trunc, k
    
    "Method to compute coefficients for truncated POD modes"
    def encoding(self, phi_k):

        a = torch.matmul(phi_k.T, self.snapshots.T) 
        return a

    "Method to perform a linear backmapping to high-dimensional space"
    def backmapping(self, a):

        field = torch.matmul(self.phi_trunc.to(**tkwargs), a.mT)
        return field.mT*self.std + self.mean

class AutoencoderReduction(DimensionalityReduction):

    def __init__(self, S, nn_model): 

         self._setsnapshots(S)
         # Setting the neural network model used for dimensionality reduction
         self.model = nn_model
         if torch.cuda.is_available():
            self.model.cuda()

    "Method to fit the PyTorch neural network model"
    def fit(self, epochs, verbose = False):

        # Training autoencoder model
        loss_function = torch.nn.MSELoss()
        print(self.snapshots.shape)
        optimizer = torch.optim.Adam(self.model.parameters(),
                                    lr = 1e-3,
                                    weight_decay = 1e-8)
        epochs = epochs
        losses = []
        for epoch in range(epochs):
            
            # Output of Autoencoder
            reconstructed = self.model(self.snapshots)
            
            # Calculating the loss function
            loss = loss_function(reconstructed, self.snapshots)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if verbose:
                print('Epoch: ', epoch, 'Loss: ', loss.item())
            
            # Storing losses for printing
            losses.append(loss.item())

    "Method to compute the encoding using the encoder neural network"
    def encoding(self):

        return self.model.encoder(self.snapshots)

    "Method to compute high dimensional solution using the decoder neural network"
    def backmapping(self, z):

        return self.model.decoder(z)
