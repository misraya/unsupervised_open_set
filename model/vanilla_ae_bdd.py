import torch
import torch.nn as nn
from model.utils import clamp_to_unit_sphere
#from utils import clamp_to_unit_sphere


class Encoder(nn.Module):
    def __init__(self, latent_size=100):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(3,64,3,1,1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64,64,3,1,1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64,128,3,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Dropout2d(0.2),
            nn.Conv2d(128,128,3,1,1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128,128,3,1,1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128,128,3,2,1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),   
        )

        self.block2 = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(128,128,3,1,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128,128,3,1,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128,128,3,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.block3 = nn.Sequential(
            nn.Dropout2d(0.2), 
            nn.Conv2d(128,128,3,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.block1_out = nn.Conv2d(128,latent_size,3,1,1,bias=False)
        self.block2_out = nn.Conv2d(128,latent_size,3,1,1,bias=False)
        self.block3_out = nn.Conv2d(128,latent_size,3,1,1,bias=False)

        self.linear = nn.Linear(128*45*80, latent_size) # 128*2*2 = 512

    def forward(self, x, out_scale=1):

        out = self.block1(x)

        if out_scale == 8:
            out = self.block1_out(out).view(out.shape[0],-1)
            return clamp_to_unit_sphere(out, out_scale*out_scale)
        
        out = self.block2(out)

        if out_scale == 4:
            out = self.block2_out(out).view(out.shape[0],-1)
            return clamp_to_unit_sphere(out, out_scale*out_scale)

        out = self.block3(out)

        if out_scale == 2:
            out = self.block3_out(out).view(out.shape[0],-1)
            return clamp_to_unit_sphere(out, out_scale*out_scale)
        
        out = self.linear(out.view(out.shape[0],-1))
        
        return clamp_to_unit_sphere(out)
    
class Decoder(nn.Module):
    def __init__(self, latent_size=100):
        super().__init__()

        self.latent_size = latent_size
        self.linear = nn.Linear(latent_size, 512*45*80, bias=False) #512*2*2

        self.block1_in = nn.ConvTranspose2d(latent_size,512,1,1,0,bias=False)
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(512,512,4,2,1,bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
        )

        self.block2_in = nn.ConvTranspose2d(latent_size,512,1,1,0,bias=False)
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(512,256,4,2,1,bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
        )

        self.block3_in = nn.ConvTranspose2d(latent_size,256,1,1,0,bias=False)
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1,bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
        )

        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(128,3,4,2,1),
            nn.Sigmoid()
        )
        


    def forward(self, x, in_scale=1):

        if in_scale <= 1:
            out = self.linear(x).reshape(-1,512,45,80)

        if in_scale == 2:
            out = out.view(-1, self.latent_size,in_scale,in_scale)
            out = self.block1_in(out)
        if in_scale <= 2:
            out = self.block1(out)

        if in_scale == 4:
            out = out.view(-1, self.latent_size,in_scale,in_scale)
            out = self.block2_in(out)
        if in_scale <= 4:
            out = self.block2(out)

        if in_scale == 8:
            out = out.view(-1, self.latent_size,in_scale,in_scale)
            out = self.block3_in(out)
        if in_scale <= 8:
            out = self.block3(out)

        return self.block4(out)
    
class VanillaAE(nn.Module):
    def __init__(self, latent_size=128):
        super().__init__()

        self.latent_size = latent_size
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)

    def forward(self, x):

        return self.decoder(self.encoder(x))
    
class VAE(nn.Module):
    def __init__(self, latent_size=128):
        super().__init__()

        self.latent_size = latent_size
        self.encoder = Encoder(2*latent_size)
        self.decoder = Decoder(latent_size)

    def forward(self, x):
        
        zus = self.encoder(x)
        b = zus.shape[0]
        mu, std = zus.chunk(2,dim=1)
        latent = std.exp() * torch.randn(zus.shape[0], zus.shape[1]//2).to(zus.device) + mu
        return self.decoder(latent), zus
    
    def generate(self, x):
        zus = self.encoder(x)
        b = zus.shape[0]
        mu, std = zus.chunk(2,dim=1)
        latent = mu
        return self.decoder(latent)
