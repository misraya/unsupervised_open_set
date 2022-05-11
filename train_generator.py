import os
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
from torchvision.utils import make_grid, save_image

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 32, 32)
    return x

def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    train_losses = []

    for batch, _ in tqdm(dataloader):
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = loss_fn(batch, model(batch))
        loss = loss.sum((1,2,3)).mean(0)
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
            loss = loss.sum((1,2,3)).mean(0)
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
        "epochs":100
    }

    wandb.init(project="547_term", config=configs)
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VanillaAE().to(device)
    loss_fn = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=config.betas)

    transform = T.Compose([T.Resize(32), T.ToTensor(), ])
    train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=2)

    val_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=2)

    train_losses = []
    val_losses = []

    for i in range(config.epochs):
        train_loss = train(model, train_loader, optimizer,loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)

        wandb.log({"train loss":sum(train_loss)/len(train_loss), "val loss":val_loss})

        train_losses.extend(train_loss)
        val_losses.append(val_loss)

        print('epoch [{}/{}], loss:{:.4f}'.format(i+1, config.epochs, sum(train_loss)/len(train_loss)))

        imgs = next(iter(val_loader))[0]
        out = model(imgs.to(device))
        recons = to_img(out.cpu().data)
        org = to_img(imgs.cpu().data)
        all = torch.stack((org,recons),dim=1).view(-1,3,32,32)
        print(all.shape)
        save_image(all, './plots/vanilla_recons_epoch{}.png'.format(i))

        if (i+1) % 5 == 0:
            torch.save(model, "ckpt/vanilla_ae/checkpoint_"+str(i)+".pth")


if __name__ == "__main__":
    main()
