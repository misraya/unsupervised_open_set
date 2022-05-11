# Adapted from https://github.com/lwneal/counterfactual-open-set/blob/master/generativeopenset/network_definitions.py
import torch
import torch.nn as nn

from model.utils import clamp_to_unit_sphere


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

        self.linear = nn.Linear(128*2*2, latent_size)

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