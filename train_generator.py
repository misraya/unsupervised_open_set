from distutils.command.config import config
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
from model.vanilla_ae import VanillaAE
import wandb

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)


def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    train_losses = []

    for batch, _ in tqdm(dataloader):
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = loss_fn(batch, model(batch))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    return train_losses


def evaluate(model, dataloader, loss_fn, device):

    model.eval()
    eval_losses = []

    with torch.no_grad():
        for batch, _ in tqdm(dataloader):
            batch = batch.to(device)
            loss = loss_fn(batch, model(batch))
            eval_losses.append(loss.item())
    
    return sum(eval_losses) / len(eval_losses)
    

def plot_losses(train_loss, val_loss):

    epochs = range(1, 1+len(train_loss[0]))

    plt.plot(epochs, train_loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='validation loss')
    plt.title('Vanilla AE Recons Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plots/vanilla_ae_recons.png', bbox_inches='tight')
    plt.close()


def main():

    configs = {
        "batch_size": 64,
        "learning_rate":1e-3,
        "betas":(0.5, 0.999),
        "iters":1000
    }

    wandb.init(project="547_term", config=configs)
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VanillaAE().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=config.betas)
    epochs = config.iters // config.batch_size


    transform = T.Compose([T.Resize(32), T.ToTensor(), ])
    train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2)

    val_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=16, shuffle=False, num_workers=2)


    train_losses = []
    val_losses = []

    for i in range(epochs):
        train_loss = train(model, train_loader, optimizer,loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)

        wandb.log({"train loss":sum(train_loss)/len(train_loss), "val loss":val_loss})

        train_losses.extend(train_loss)
        val_losses.append(val_loss)

    

if __name__ == "__main__":
    main()
