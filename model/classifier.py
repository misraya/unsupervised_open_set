# Adapted from https://github.com/lwneal/counterfactual-open-set/blob/master/generativeopenset/network_definitions.py
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(6,64,3,1,1,bias=False),
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

        self.linear = nn.Linear(128*4*4, num_classes)


    def forward(self, x, return_features=False):

        out = self.layers(x).view(x.shape[0],-1)

        if return_features:
            return out
        
        return self.linear(out)
