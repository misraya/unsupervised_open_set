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
from model.wgan import SAVE_PER_TIMES, WGAN_GP
import wandb
from torchvision.utils import make_grid, save_image
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import time as t

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
SAVE_PER_TIMES=50

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


def train(model, train_loader, config):
    model.t_begin = t.time()
    model.file = open("inception_score_graph.txt", "w")

    # Now batches are callable model.data.next()
    model.data = model.get_infinite_batches(train_loader)

    one = torch.tensor(1, dtype=torch.float)
    mone = one * -1
    if model.cuda:
        one = one.cuda(model.cuda_index)
        mone = mone.cuda(model.cuda_index)

    for g_iter in range(model.generator_iters):
        # Requires grad, Generator requires_grad = False
        for p in model.D.parameters():
            p.requires_grad = True

        d_loss_real = 0
        d_loss_fake = 0
        Wasserstein_D = 0
        # Train Dicriminator forward-loss-backward-update model.critic_iter times while 1 Generator forward-loss-backward-update
        for d_iter in range(model.critic_iter):
            model.D.zero_grad()

            images = model.data.__next__()
            # Check for batch to have full batch_size
            if (images.size()[0] != config.batch_size):
                continue

            z = torch.rand((config.batch_size, 100, 1, 1))

            images, z = model.get_torch_variable(images), model.get_torch_variable(z)

            # Train discriminator
            # WGAN - Training discriminator more iterations than generator
            # Train with real images
            d_loss_real = model.D(images)
            d_loss_real = d_loss_real.mean()
            d_loss_real.backward(mone)

            # Train with fake images
            z = model.get_torch_variable(torch.randn(config.batch_size, 100, 1, 1))

            fake_images = model.G(z)
            d_loss_fake = model.D(fake_images)
            d_loss_fake = d_loss_fake.mean()
            d_loss_fake.backward(one)

            # Train with gradient penalty
            gradient_penalty = model.calculate_gradient_penalty(images.data, fake_images.data)
            gradient_penalty.backward()


            d_loss = d_loss_fake - d_loss_real + gradient_penalty
            Wasserstein_D = d_loss_real - d_loss_fake
            model.d_optimizer.step()
            print(f'  Discriminator iteration: {d_iter}/{model.critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

        # Generator update
        for p in model.D.parameters():
            p.requires_grad = False  # to avoid computation

        model.G.zero_grad()
        # train generator
        # compute loss with fake images
        z = model.get_torch_variable(torch.randn(config.batch_size, 100, 1, 1))
        fake_images = model.G(z)
        g_loss = model.D(fake_images)
        g_loss = g_loss.mean()
        g_loss.backward(mone)
        g_cost = -g_loss
        model.g_optimizer.step()
        print(f'Generator iteration: {g_iter}/{model.generator_iters}, g_loss: {g_loss}')
        # Saving model and sampling images every 100th generator iterations
        if (g_iter) % SAVE_PER_TIMES == 0:
            model.save_model()
            # # Workaround because graphic card memory can't store more than 830 examples in memory for generating image
            # # Therefore doing loop and generating 800 examples and stacking into list of samples to get 8000 generated images
            # # This way Inception score is more correct since there are different generated examples from every class of Inception model
            # sample_list = []
            # for i in range(125):
            #     samples  = model.data.__next__()
            # #     z = Variable(torch.randn(800, 100, 1, 1)).cuda(model.cuda_index)
            # #     samples = model.G(z)
            #     sample_list.append(samples.data.cpu().numpy())
            # #
            # # # Flattening list of list into one list
            # new_sample_list = list(chain.from_iterable(sample_list))
            # print("Calculating Inception Score over 8k generated images")
            # # # Feeding list of numpy arrays
            # inception_score = get_inception_score(new_sample_list, cuda=True, batch_size=32,
            #                                       resize=True, splits=10)

            if not os.path.exists('training_result_images/'):
                os.makedirs('training_result_images/')

            # Denormalize images and save them in grid 8x8
            z = model.get_torch_variable(torch.randn(800, 100, 1, 1))
            samples = model.G(z)
            samples = samples.mul(0.5).add(0.5)
            samples = samples.data.cpu()[:64]
            grid = make_grid(samples)
            save_image(grid, 'training_result_images/img_generatori_iter_{}.png'.format(str(g_iter).zfill(3)))

            # Testing
            time = t.time() - model.t_begin
            #print("Real Inception score: {}".format(inception_score))
            print("Generator iter: {}".format(g_iter))
            print("Time {}".format(time))

            # Write to file inception_score, gen_iters, time
            #output = str(g_iter) + " " + str(time) + " " + str(inception_score[0]) + "\n"
            #model.file.write(output)

            real = model.real_images(images, model.number_of_images)
            generated = model.generate_img(z, model.number_of_images)
            merged = torch.stack((real, generated), dim=1).view(-1,3,32,32)

            info = {
                'Wasserstein distance': Wasserstein_D.data,
                'Loss D': d_loss.data,
                'Loss G': g_cost.data,
                'Loss D Real': d_loss_real.data,
                'Loss D Fake': d_loss_fake.data,
                'images': wandb.Image(merged, caption="images (real T, generated B)"),
            }

            wandb.log(info)


    model.t_end = t.time()
    print('Time of training-{}'.format((model.t_end - model.t_begin)))
    #model.file.close()

    # Save the trained parameters
    model.save_model()


def evaluate(model, test_loader, config, D_model_path="ckpt/discriminator.pkl", G_model_path="ckpt/generator.pkl"):
    model.load_model(D_model_path, G_model_path)
    z = model.get_torch_variable(torch.randn(config.batch_size, 100, 1, 1))
    samples = model.G(z)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.data.cpu()
    grid = make_grid(samples)
    print("Grid of 8x8 images saved to 'wgan_model_image.png'.")
    save_image(grid, 'wgan_model_image.png')
    


def main():


    configs = {
        "model":"wgan_gp",
        "channels":3,
        "batch_size": 64,
        "learning_rate":1e-3,
        "betas":(0.5, 0.999),
        "iters": 1000,
        "train": True,
        "cuda": True,
        "split": 1
    }

    wandb.init(project="547_term", config=configs)
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WGAN_GP(config).to(device)
    
    train_transform = T.Compose([T.ToPILImage(), T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    val_transform = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True)
    val_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True)

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

    if config.train:
        train(model,train_loader, config)

    else:
        evaluate(model,val_loader, config)
        for i in range(50):
            model.generate_latent_walk(i)

if __name__ == "__main__":
    main()
