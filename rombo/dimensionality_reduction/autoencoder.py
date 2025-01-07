"""
Definition of standard MLP autoencoder with a symmetric architecture

Other custom models can be defined but should have a decoder and encoder network within the definition

"""

import torch.nn as nn
import torch

def init_weights(m):

    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

class MLPAutoEnc(nn.Module):

    "Definition of an MLP autoencoder"
    def __init__(self, high_dim, hidden_dims, activation, zd):
        super(MLPAutoEnc, self).__init__()

        encoder_layers = []
        last_dim = high_dim

        for dim in hidden_dims:

            encoder_layers.append(nn.Linear(last_dim, dim))
            encoder_layers.append(activation)
            last_dim = dim
        
        encoder_layers.append(nn.Linear(last_dim, zd))

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        last_dim = zd
        hidden_dims.reverse()
        for dim in hidden_dims:

            decoder_layers.append(nn.Linear(last_dim, dim))
            decoder_layers.append(activation)
            last_dim = dim

        decoder_layers.append(nn.Linear(last_dim, high_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):

        encoded = self.encoder(x)
        return self.decoder(encoded)










