import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder

class VanillaAE(nn.Module):
    def __init__(self, latent_size=100):
        super().__init__()

        self.latent_size = latent_size
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)

    def forward(self, x):

        return self.decoder(self.encoder(x))
