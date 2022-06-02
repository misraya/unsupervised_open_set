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
from model.wgan import WGAN_GP

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

CIFAR_CLASSES = ['plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
SPLITS = [ [3, 6, 7, 8],
        [1, 2, 4, 6],
        [2, 3, 4, 9],
        [0, 1, 2, 6],
        [4, 5, 6, 9]]
KNOWN_SPLITS = [[0,1,2,4,5,9],
        [0,3,5,7,8,9],
        [0,1,5,6,7,8],
        [3,4,5,7,8,9],
        [0,1,2,3,7,8]]
KNOWN_SPLIT_NAMES = [[CIFAR_CLASSES[i] for i in s] for s in KNOWN_SPLITS]


def train(G, C, dataloader, optimizer, loss_fn, transformations, device):
    C.train()

    train_losses = []

    for x, y in tqdm(dataloader):
        x = x.to(device)
        x_hat = G(x)
        concat_x = torch.cat((x, x_hat), dim=1)
        ce_loss = loss_fn(C(concat_x)[0], y)

        trans_ind = torch.randint(len(transformations), (1,))
        t_x = transformations[trans_ind](x)
        t_x_hat = transformations[trans_ind](x_hat)
        concat_t = torch.cat((t_x, t_x_hat), dim=1)
        ss_loss = loss_fn(C(concat_t)[1], trans_ind)

        loss = 0.8 * ce_loss + 0.2 * ss_loss

        optimizer.zero_grad()
        loss = loss.sum((1,2,3)).mean(0)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    return train_losses


def evaluate(G, C, dataloader, loss_fn, KNOWN_CLASSES, UNK_INDEX, device):
    C.eval()
    eval_losses, acc = []

    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x = x.to(device)
            x_hat = G(x)
            concat_x = torch.cat((x, x_hat), dim=1)
            pred = F.softmax(C(concat_x)[0])
            max_act, indices = torch.max(pred, dim=-1)
            pred = torch.where(max_act < 0.90, UNK_INDEX, pred)
            unk_y = torch.where(torch.isin(y, KNOWN_CLASSES), y, UNK_INDEX)

            ce_loss = loss_fn(pred, unk_y)

            eval_losses.append(ce_loss.item())
            acc.append(sum(pred == unk_y))
    
    return sum(eval_losses) / len(eval_losses), sum(acc / len(acc))



# TO BE COMPLETED
def main():

    configs = {
        "batch_size": 64,
        "learning_rate":1e-3,
        "betas":(0.5, 0.999),
        "epochs": 250,
        "split": 1
        "unk_index": 11
        "type": "train open-set-classifier"
    }

    wandb.init(project="547_term", config=configs)
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = T.Compose([T.ToPILImage(), T.RandomCrop(32, padding=4), T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    val_transform = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True)
    val_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transforms=val_transform)

    # Separating trainset/testset data/label
    x_train = train_set.data
    y_train = train_set.targets

    split_train_set = DatasetMaker(
        [get_class_i(x_train, y_train, classDict[class_name]) for class_name in KNOWN_SPLIT_NAMES[config.split]], train_transform)

    # Create datasetLoaders from trainset and testset
    train_loader = DataLoader(split_train_set, batch_size=config.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=8)

    model = VanillaAE().to(device)
    generator = torch.load("ckpt/norm_vanilla_ae_split1/checkpoint_249.pth").to(device)
    classifier = Classifier(num_classes=6,num_transformations=8).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=config.betas, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, threshold_mode='abs')

    # freeze generator
    for param in generator.parameters():
        param.requires_grad = False

    # 8 distinct outputs
    transformations = [
        T.RandomRotation(degrees=[90,90]), # deterministic rotation
        T.RandomRotation(degrees=[180,180]),
        T.RandomRotation(degrees=[270,270]),
        T.RandomRotation(degrees=[360,360]),
        T.Compose([T.RandomHorizontalFlip(p=1.), T.RandomRotation(degrees=[90,90])]), # deterministic flip + rotation
        T.Compose([T.RandomHorizontalFlip(p=1.), T.RandomRotation(degrees=[180,180])]),
        T.Compose([T.RandomHorizontalFlip(p=1.), T.RandomRotation(degrees=[270,270])]),
        T.Compose([T.RandomHorizontalFlip(p=1.), T.RandomRotation(degrees=[360,360])]),
    ]


    for i in range(config.epochs):
        train_loss = train(generator, classifier, train_loader, optimizer, loss_fn, transformations, device)
        val_loss, val_acc = evaluate(generator, classifier, val_loader, loss_fn, KNOWN_SPLITS[config.split], config.unk_index, device)
        scheduler.step(val_loss)

        print('epoch [{}/{}], lr:{:.4f}, train loss:{:.4f}, val loss:{:.4f}, val-open set acc:{:.4f}'.format(i+1, config.epochs, optimizer.param_groups[0]['lr'], sum(train_loss)/len(train_loss), val_loss, val_acc))
        wandb.log({"train loss":sum(train_loss)/len(train_loss), "val loss":val_loss, "lr":optimizer.param_groups[0]['lr'], "val - open set accuracy": val_acc})


if __name__ == "__main__":
    main()
