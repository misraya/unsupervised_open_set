import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
import wandb
from model.vanilla_ae import VanillaAE
from model.classifier import Classifier

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

def train(G, C, dataloader, optimizer, loss_fn, transformations, device):
    C.train()

    train_losses = []

    for x, y in tqdm(dataloader):
        x = x.to(device)
        x_hat = G(x)
        concat_x = torch.cat((x, x_hat), dim=1)
        ce_loss = loss_fn(C(concat_x), y)

        trans_ind = torch.randint(len(transformations))
        t_x = transformations[trans_ind](x)
        t_x_hat = transformations[trans_ind](x_hat)
        concat_t = torch.cat((t_x, t_x_hat), dim=1)
        ss_loss = loss_fn(C(concat_t), trans_ind)

        loss = 0.8 * ce_loss + 0.2 * ss_loss

        optimizer.zero_grad()
        loss = loss.sum((1,2,3)).mean(0)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    return train_losses


def evaluate(G, C, dataloader, loss_fn, device):
    C.eval()
    eval_losses = []

    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x = x.to(device)
            x_hat = G(x)
            concat_x = torch.cat((x, x_hat), dim=1)
            ce_loss = loss_fn(C(concat_x), y)
            eval_losses.append(ce_loss.item())
    
    return sum(eval_losses) / len(eval_losses)


# TO BE COMPLETED
def main():

    configs = {
        
    }

    wandb.init(project="547_term", config=configs)
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if __name__ == "__main__":
    main()
