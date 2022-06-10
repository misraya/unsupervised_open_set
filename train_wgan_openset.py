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
from torchmetrics import AUROC

from model.vanilla_ae import VanillaAE
from model.classifier import Classifier
from model.wgan import WGAN_GP
from model.utils import to_img, to_4d
from data.dataset_maker import split_dataset, KNOWN_SPLITS, KNOWN_SPLIT_NAMES, CIFAR_CLASSES

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)


# train on known classes, both classification and self supervision
def train(encoder, generator, classifier, dataloader, optimizer, loss_fn, transformations, device):
    encoder.train()
    generator.train()
    classifier.train()

    ce_losses, ss_losses, train_losses = [], [], []

    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)
        x_hat = generator.G(to_4d(encoder(x)))
        concat_x = torch.cat((x, x_hat), dim=1)
        ce_loss = loss_fn(classifier(concat_x)[0], y)

        # note: how to get rid of for loop
        trans_ind = torch.randint(len(transformations), (x.size(0),))
        rand_trans = transformations[trans_ind]
        t_x = torch.stack([t(x[i]) for i,t in enumerate(rand_trans)], dim=0)
        t_x_hat = torch.stack([t(x_hat[i]) for i,t in enumerate(rand_trans)], dim=0)

        concat_t = torch.cat((t_x, t_x_hat), dim=1)
        ss_loss = loss_fn(classifier(concat_t)[1], trans_ind.to(device))

        loss = 0.8 * ce_loss + 0.2 * ss_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ce_losses.append(ce_loss.item())
        ss_losses.append(ss_loss.item())
        train_losses.append(loss.item())

    return ce_losses, ss_losses, train_losses


# validate on known classes, find threshold value where min TP is 0.90
def find_threshold_at_tp_level(encoder, generator, classifier, val_loader, ce_loss, KNOWN_CLASSES, CIFAR_CLASSES, device, min_tp=0.90):
    encoder.eval()
    generator.eval()
    classifier.eval()

    preds, targets = [], []

    # collect preds and GT for the validation set
    with torch.no_grad():
        for x, y in tqdm(val_loader):
            x, y = x.to(device), y.to(device)
            x_hat = generator.G(to_4d(encoder(x)))
            concat_x = torch.cat((x, x_hat), dim=1)
            logits = classifier(concat_x)[0]
            probs = F.softmax(logits, dim=-1)
            preds.append(probs)
            targets.append(y)

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    val_ce = ce_loss(preds, targets)

    pred_max, pred_idx = torch.max(preds, 1)
    val_acc = (pred_idx==targets).sum().item() / len(targets)

    wandb.log({"val - PR" : wandb.plot.pr_curve(targets.cpu(), preds.cpu(), labels=CIFAR_CLASSES), "val - ROC": wandb.plot.roc_curve(targets.cpu(), preds.cpu(), labels=CIFAR_CLASSES)})

    # calculate FPR and TPR rates
    fpr, tpr, thr, roc_auc = dict(), dict(), dict(), dict()
    for i, cl in enumerate(KNOWN_CLASSES):
        fpr[i], tpr[i], thr[i] = roc_curve((targets==i).cpu(), preds[:,i].cpu())
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    thresholds = []
    for i, cl in enumerate(KNOWN_CLASSES):
        thr_idx = np.argmax(tpr[i]>=min_tp)
        thresholds.append(thr[i][thr_idx])

    return val_ce, val_acc, np.max(thresholds)


# evaluate on full dataset that consists of both known and unknown classes
def evaluate(encoder, generator, classifier, dataloader, threshold, KNOWN_CLASSES, UNK_INDEX, device, vis=False):
    encoder.eval()
    generator.eval()
    classifier.eval()
    acc = 0.0
    acc2 = 0.0
    auroc_preds, auroc_preds_2d, auroc_targets = [], [], []

    test_data_at = wandb.Artifact("evaluate" + str(wandb.run.id), type="predictions")
    columns = ["image","gt_class","unk_y","bin_y","max_act","threshold","prob_known","prob_unk"]
    test_table = wandb.Table(columns=columns)

    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            x_hat = generator.G(to_4d(encoder(x)))
            concat_x = torch.cat((x, x_hat), dim=1)
            logits = classifier(concat_x)[0]
            probs = F.softmax(logits, dim=-1)

            max_act, indices = torch.max(probs, dim=-1)
            labels = torch.where(max_act < threshold, UNK_INDEX, indices)
            unk_y = torch.where(torch.isin(y, torch.Tensor(KNOWN_CLASSES).to(device)), y, UNK_INDEX)
            
            acc += (labels == unk_y).sum().item()


            labels2 = torch.where(max_act < threshold, 1, 0)
            unk_y2 = torch.where(torch.isin(y, torch.Tensor(KNOWN_CLASSES).to(device)), 0, 1)
            acc2 += (labels2 == unk_y2).sum().item()



            # The implicit K+1th class (the open set class) is computed
            # by assuming an extra linear output with constant value 0
            # https://github.com/lwneal/counterfactual-open-set/blob/34fbc726fb7fe76d15fb323e9597c76292b66d81/generativeopenset/evaluation.py#L217
            z = torch.exp(logits).sum(dim=1)
            prob_known = z / (z + 1)
            prob_unknown = 1 - prob_known
            bin_y = torch.where(torch.isin(y, torch.Tensor(KNOWN_CLASSES).to(device)), 0, 1) # unk -> 1, known -> 0
            auroc_preds.append(prob_unknown)
            auroc_preds_2d.append(torch.stack([prob_known, prob_unknown], dim=1))
            auroc_targets.append(bin_y)
            
            test_table.add_data(
                wandb.Image(to_img(x[0:1].cpu().data)), 
                y[0].cpu().data, 
                unk_y[0].cpu().data,
                bin_y[0].cpu().data,
                max_act[0].cpu().data,
                threshold,
                prob_known[0].cpu().data,
                prob_unknown[0].cpu().data,
                )    


    auroc_preds = torch.cat(auroc_preds, dim=0)
    auroc_preds_2d = torch.cat(auroc_preds_2d, dim=0)
    auroc_targets = torch.cat(auroc_targets, dim=0)

    wandb.log({"open set - PR" : wandb.plot.pr_curve(auroc_targets.cpu(), auroc_preds_2d.cpu(), labels=["known","unknown"]), 
                "open set - ROC": wandb.plot.roc_curve(auroc_targets.cpu(), auroc_preds_2d.cpu(), labels=["known","unknown"]),
            })
    test_data_at.add(test_table, "predictions")
    wandb.run.log_artifact(test_data_at)      

    mean_acc = acc / len(dataloader.dataset)

    # print("binary accuracy", acc2 / len(dataloader.dataset))

    auroc = AUROC(pos_label=1)(auroc_preds, auroc_targets)
    return  mean_acc, auroc.item()


def main():

    configs = {
        "channels":3,
        "train": False,
        "cuda": True,
        "iters": 40000,
        "batch_size": 64,
        "learning_rate":1e-3,
        "betas":(0.5, 0.999),
        "epochs": 150,
        "split": 1,
        "unk_index": 11,
        "ckpt_period":25,
        "generator":"wgan",
        "classifier":"vanilla_cnn",
        "type": "train open-set-classifier",
        "eval_only":False
    }

    wandb.init(project="547_term", config=configs)
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join("ckpt/openset_wgan_cnn", "cifar_split" + str(config.split))
    out_path = os.path.join("output/openset_wgan_cnn", "cifar_split" + str(config.split))
    config.out_path = out_path
    config.ckpt_path = ckpt_path

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

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
    train_loader = DataLoader(split_train_set, batch_size=config.batch_size, shuffle=False, num_workers=8, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(split_train_set, batch_size=config.batch_size, shuffle=False, num_workers=8, sampler=val_sampler, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=8)


    encoder = torch.load("ckpt/wgan_encoder/cifar_split"+str(config.split)+"/ckpt100.pth")
    generator = WGAN_GP(config).to(device)
    d_path = "ckpt/wgan_gp/cifar_split"+str(config.split)+"/discriminator.pkl"
    g_path = "ckpt/wgan_gp/cifar_split"+str(config.split)+"/generator.pkl"
    generator.load_model(d_path, g_path)

    # freeze encoder and wgan
    for param in list(generator.G.parameters()) + list(generator.D.parameters()) + list(encoder.parameters()):
        param.requires_grad = False

    classifier = Classifier(num_classes=6,num_transformations=8).to(device)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=config.learning_rate, betas=config.betas, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=10, threshold_mode='abs')

    # 8 distinct outputs
    transformations = np.array([
        T.RandomRotation(degrees=[90,90]), # deterministic rotation
        T.RandomRotation(degrees=[180,180]),
        T.RandomRotation(degrees=[270,270]),
        T.RandomRotation(degrees=[360,360]), # original input
        T.Compose([T.RandomHorizontalFlip(p=1.), T.RandomRotation(degrees=[90,90])]), # deterministic flip + rotation
        T.Compose([T.RandomHorizontalFlip(p=1.), T.RandomRotation(degrees=[180,180])]),
        T.Compose([T.RandomHorizontalFlip(p=1.), T.RandomRotation(degrees=[270,270])]),
        T.Compose([T.RandomHorizontalFlip(p=1.), T.RandomRotation(degrees=[360,360])]),
    ])

    if config.eval_only:
        classifier = torch.load("ckpt/openset_wgan_cnn/cifar_split"+str(config.split)+"/ckpt150.pth").to(device)

        val_ce, val_acc, threshold = find_threshold_at_tp_level(encoder, generator, classifier, val_loader, ce_loss, KNOWN_SPLITS[config.split], CIFAR_CLASSES, device)
        test_acc, test_auroc = evaluate(encoder, generator, classifier, test_loader, threshold, KNOWN_SPLITS[config.split], config.unk_index, device)

        print('val loss:{:.4f}, closed set (val) acc:{:.4f}, open set (test) acc:{:.4f}, open set (test) auroc:{:.4f}'.format(val_ce, val_acc, test_acc, test_auroc))
        wandb.log({"val - ce loss": val_ce, 
                    "val - closed set accuracy": val_acc, 
                    "test - open set accuracy": test_acc,
                    "test - auroc": test_auroc,
                    })
        return

    for i in range(config.epochs):
        train_ce_loss, train_ss_loss, train_loss = train(encoder, generator, classifier, train_loader, optimizer, ce_loss, transformations, device)
        val_ce, val_acc, threshold = find_threshold_at_tp_level(encoder, generator, classifier, val_loader, ce_loss, KNOWN_SPLITS[config.split], CIFAR_CLASSES, device)
        test_acc, test_auroc = evaluate(encoder, generator, classifier, test_loader, threshold, KNOWN_SPLITS[config.split], config.unk_index, device)
        scheduler.step(val_ce)

        print('epoch [{}/{}], lr:{:.4f}, train loss:{:.4f}, val loss:{:.4f}, closed set (val) acc:{:.4f}, open set (test) acc:{:.4f}, open set (test) auroc:{:.4f}'.format(i+1, config.epochs, optimizer.param_groups[0]['lr'], sum(train_loss)/len(train_loss), val_ce, val_acc, test_acc, test_auroc))
        wandb.log({"train - ce loss": sum(train_ce_loss)/len(train_ce_loss),
                    "train - ss loss": sum(train_ss_loss)/len(train_ss_loss),
                    "train - total loss": sum(train_loss)/len(train_loss),
                    "lr": optimizer.param_groups[0]['lr'], 
                    "val - ce loss": val_ce, 
                    "val - closed set accuracy": val_acc, 
                    "test - open set accuracy": test_acc,
                    "test - auroc": test_auroc,
                    })
        if (i + 1) % config.ckpt_period == 0:
            save_path = os.path.join(config.ckpt_path, "ckpt"+str(i+1)+".pth")
            torch.save(classifier, save_path)

if __name__ == "__main__":
    main()
