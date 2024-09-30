"""

Definition of classes for dimensionality reduction using linear and nonlinear methods

- Proper Orthogonal Decomposition
- Manifold Learning - Torch implementation might be difficult here
- Autoencoders

"""
import torch
from abc import ABC, abstractmethod

class DimensionalityReduction(ABC):

    "Method to assign snapshot vectors and center them if required"
    def _setsnapshots(self, S, center = True):

        self.snapshots = S 
        # Add in centering capability

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

    def __init__(self, S, RIC):

        # Relative information content is the only parameter required
        self.ric = RIC
        self._setsnapshots(S)

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
        phi_trunc = phi[:,0:k]
        
        return phi_trunc, k
    
    "Method to compute coefficients for truncated POD modes"
    def encoding(self, phi_k):

        a = torch.matmul(phi_k.T, self.snapshots.T) 
        return a

    "Method to perform a linear backmapping to high-dimensional space"
    def backmapping(self, phi_trunc, a):

        field = torch.matmul(phi_trunc, a.T)
        return field

class AutoencoderReduction(DimensionalityReduction):

    def __init__(self, S, nn_model): 

         self._setsnapshots(S)
         # Setting the neural network model used for dimensionality reduction
         self.model = nn_model

    "Method to fit the PyTorch neural network model"
    def fit(self, epochs):

        # Training autoencoder model
        loss_function = torch.nn.MSELoss()

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
            
            # Storing losses for printing
            losses.append(loss.item())

    "Method to compute the encoding using the encoder neural network"
    def encoding(self):

        return self.model.encoder(self.snapshots)

    "Method to compute high dimensional solution using the decoder neural network"
    def backmapping(self, z):

        return self.model.decoder(z)
