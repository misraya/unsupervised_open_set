import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas
import random
import wandb
from collections import Counter
from torchvision.utils import make_grid, save_image
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from sklearn.metrics import roc_curve, roc_auc_score, auc

from model.vanilla_ae import VanillaAE
from model.classifier import Classifier
from model.wgan import WGAN_GP
from model.utils import to_img
from data.dataset_maker import split_dataset, KNOWN_SPLITS, KNOWN_SPLIT_NAMES, CIFAR_CLASSES

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)


def train(G, C, dataloader, optimizer, loss_fn, transformations, device):
    C.train()

    train_losses = []

    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)
        x_hat = G(x)
        concat_x = torch.cat((x, x_hat), dim=1)
        ce_loss = loss_fn(C(concat_x)[0], y)

        trans_ind = torch.randint(len(transformations), (1,))
        t_x = transformations[trans_ind](x)
        t_x_hat = transformations[trans_ind](x_hat)
        concat_t = torch.cat((t_x, t_x_hat), dim=1)
        ss_loss = loss_fn(C(concat_t)[1], trans_ind.repeat(len(x)).to(device))

        loss = 0.8 * ce_loss + 0.2 * ss_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    return train_losses

# validation stage
def find_threshold_at_tp_level(G, C, val_loader, ce_loss, KNOWN_CLASSES, CIFAR_CLASSES, device, min_tp=0.90):
    C.eval()

    preds, targets = [], []

    # collect preds and GT for the validation set
    with torch.no_grad():
        for x, y in tqdm(val_loader):
            x, y = x.to(device), y.to(device)
            x_hat = G(x)
            concat_x = torch.cat((x, x_hat), dim=1)
            pred = F.softmax(C(concat_x)[0], dim=-1)
            preds.append(pred)
            targets.append(y)

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    val_ce = ce_loss(preds, targets)

    _, pred_idx = torch.max(preds, 1)
    val_acc = (pred_idx==targets).sum().item() / len(targets)

    wandb.log({"PR curve" : wandb.plot.pr_curve(targets.cpu(), preds.cpu(), labels=CIFAR_CLASSES), "ROC curve": wandb.plot.roc_curve(targets.cpu(), preds.cpu(), labels=CIFAR_CLASSES)})

    # calculate FPR and TPR rates
    fpr, tpr, thr, roc_auc = dict(), dict(), dict(), dict()
    for i, cl in enumerate(KNOWN_CLASSES):
        fpr[i], tpr[i], thr[i] = roc_curve((targets==i).cpu(), preds[:,i].cpu())
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    thresholds = []
    for i, cl in enumerate(KNOWN_CLASSES):
        thr_idx = np.argmax(tpr[i]>=0.9)
        thresholds.append(thr[i][thr_idx])

    return val_ce, val_acc, min(thresholds)


def evaluate(G, C, dataloader, threshold, KNOWN_CLASSES, UNK_INDEX, device, vis=False):
    C.eval()
    acc = 0.0

    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            x_hat = G(x)
            concat_x = torch.cat((x, x_hat), dim=1)
            pred = F.softmax(C(concat_x)[0], dim=-1)

            max_act, indices = torch.max(pred, dim=-1)
            labels = torch.where(max_act < threshold, UNK_INDEX, indices)
            unk_y = torch.where(torch.isin(y, torch.Tensor(KNOWN_CLASSES).to(device)), y, UNK_INDEX)
            acc += (labels == unk_y).sum().item()

    return  acc / len(dataloader.dataset)



# TO BE COMPLETED
def main():

    configs = {
        "batch_size": 64,
        "learning_rate":1e-3,
        "betas":(0.5, 0.999),
        "epochs": 250,
        "split": 1,
        "unk_index": 11,
        "ckpt_period":25,
        "ckpt_path":"ckpt/openset_vanilla/",
        "type": "train open-set-classifier"
    }

    wandb.init(project="547_term", config=configs)
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)

    # Prepare splitted dataset and loaders
    train_transform = T.Compose([T.ToPILImage(), T.RandomCrop(32, padding=4), T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    test_transform = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    
    train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True)
    split_train_set = split_dataset(train_set, config.split, train_transform) # extract known class samples

    # small val set needed to decide threshold
    train_size, val_size = int(np.floor(len(split_train_set) * 0.9)), int(np.floor(len(split_train_set) * 0.1))
    indices = list(range(len(split_train_set)))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    train_sampler, val_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(val_idx)

    test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(split_train_set, batch_size=config.batch_size, shuffle=False, num_workers=8, sampler=train_sampler)
    val_loader = DataLoader(split_train_set, batch_size=config.batch_size, shuffle=False, num_workers=8, sampler=val_sampler)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=8)

    generator = torch.load("ckpt/norm_vanilla_ae_split1/checkpoint_249.pth").to(device)
    classifier = Classifier(num_classes=6,num_transformations=8).to(device)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=config.learning_rate, betas=config.betas, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=10, threshold_mode='abs')

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
        train_loss = train(generator, classifier, train_loader, optimizer, ce_loss, transformations, device)
        val_ce, val_acc, threshold = find_threshold_at_tp_level(generator, classifier, val_loader, ce_loss, KNOWN_SPLITS[config.split], CIFAR_CLASSES, device)
        test_acc = evaluate(generator, classifier, test_loader, threshold, KNOWN_SPLITS[config.split], config.unk_index, device)
        scheduler.step(val_ce)

        print('epoch [{}/{}], lr:{:.4f}, train loss:{:.4f}, val loss:{:.4f}, closed set (val) acc: {:.4f}, open set acc:{:.4f}'.format(i+1, config.epochs, optimizer.param_groups[0]['lr'], sum(train_loss)/len(train_loss), val_ce, val_acc, test_acc))
        wandb.log({"train loss":sum(train_loss)/len(train_loss),"lr":optimizer.param_groups[0]['lr'], "val loss":val_ce, "val - closed set acc": val_acc, "test - open set accuracy": test_acc})
        if (i + 1) % config.ckpt_period == 0:
            save_path = os.path.join(config.ckpt_path, "epoch"+str(i+1)+".pth")
            torch.save(classifier, save_path)

if __name__ == "__main__":
    main()
