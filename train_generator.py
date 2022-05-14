import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
from model.vanilla_ae import VanillaAE
import wandb
from torchvision.utils import make_grid, save_image
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

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



# Define a function to separate CIFAR classes by class index
def get_class_i(x, y, i):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:, 0])
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]

    return x_i


class DatasetMaker(Dataset):
    def __init__(self, datasets, transforms):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        self.transforms = transforms

    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        img = self.transforms(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)

    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)

        return bin_index, index_wrt_class


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 32, 32)
    return x

def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    train_losses = []

    for batch, _ in tqdm(dataloader):
        batch = batch.float().to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(batch), batch)
        #loss = loss.sum((1,2,3)).mean(0)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    return train_losses


def evaluate(model, dataloader, loss_fn, device):

    model.eval()
    eval_losses = []

    with torch.no_grad():
        for batch, _ in tqdm(dataloader):
            batch = batch.float().to(device)
            loss = loss_fn(model(batch), batch)
            #loss = loss.sum((1,2,3)).mean(0)
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
        "batch_size": 128,
        "learning_rate":1e-3,
        "betas":(0.5, 0.999),
        "epochs": 250,
        "split": 1
        #"lr_sch_step":50,
        #"lr_sch_gamma":0.1
    }

    wandb.init(project="547_term", config=configs)
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VanillaAE().to(device)
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=config.betas, weight_decay=1e-4)
    #scheduler = StepLR(optimizer, step_size=config.lr_sch_step, gamma=config.lr_sch_gamma)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, threshold_mode='abs')

    train_transform = T.Compose([T.ToPILImage(), T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    val_transform = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True)
    #train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=8)

    val_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True)
    #val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=8)


    # Separating trainset/testset data/label
    x_train = train_set.data
    x_test = val_set.data
    y_train = train_set.targets
    y_test = val_set.targets

    split_train_set = DatasetMaker(
        [get_class_i(x_train, y_train, classDict[class_name]) for class_name in KNOWN_SPLIT_NAMES[config.split]], train_transform)

    split_test_set = DatasetMaker(
        [get_class_i(x_test, y_test, classDict[class_name]) for class_name in KNOWN_SPLIT_NAMES[config.split]], val_transform)


    # Create datasetLoaders from trainset and testset
    train_loader = DataLoader(split_train_set, batch_size=config.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(split_test_set, batch_size=config.batch_size, shuffle=False, num_workers=8)

    out_dir = "norm_vanilla_ae_split"+str(config.split)

    train_losses = []
    val_losses = []

    for i in range(config.epochs):
        train_loss = train(model, train_loader, optimizer,loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)

        train_losses.extend(train_loss)
        val_losses.append(val_loss)

        print('epoch [{}/{}], train loss:{:.4f}, val loss:{:.4f}'.format(i+1, config.epochs, sum(train_loss)/len(train_loss), val_loss))

        imgs = next(iter(val_loader))[0][:20]
        out = model(imgs.to(device))
        recons = to_img(out.cpu().data)
        org = to_img(imgs.cpu().data)
        merged = torch.stack((org,recons),dim=1).view(-1,3,32,32)
        save_image(merged, "./plots/"+out_dir+"/vanilla_recons_epoch{}.png".format(i))
        images = wandb.Image(merged, caption="Reconstructions")
        wandb.log({"train loss":sum(train_loss)/len(train_loss), "val loss":val_loss, "lr":optimizer.param_groups[0]['lr'],"reconstructions": images})

        if (i+1) % 10 == 0:
            torch.save(model, "ckpt/"+out_dir+"/checkpoint_"+str(i)+".pth")


if __name__ == "__main__":
    main()
