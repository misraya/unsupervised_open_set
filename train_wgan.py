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
import wandb
from torchvision.utils import make_grid, save_image
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import time as t

from model.wgan import SAVE_PER_TIMES, WGAN_GP
from model.utils import to_img
from data.dataset_maker import split_dataset

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)


def train(model, train_loader, config):
    model.t_begin = t.time()
    # model.file = open("inception_score_graph.txt", "w")

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

            if not os.path.exists(config.img_output_dir):
                os.makedirs(config.img_output_dir)

            # Denormalize images and save them in grid 8x8
            z = model.get_torch_variable(torch.randn(800, 100, 1, 1))
            samples = model.G(z)
            samples = samples.mul(0.5).add(0.5)
            samples = samples.data.cpu()[:64]
            grid = make_grid(samples)
            save_image(grid, os.path.join(config.img_output_dir, 'img_generator_iter_{}.png'.format(str(g_iter).zfill(3))))

            # Testing
            time = t.time() - model.t_begin
            #print("Real Inception score: {}".format(inception_score))
            print("Generator iter: {}".format(g_iter))
            print("Time {}".format(time))

            # Write to file inception_score, gen_iters, time
            #output = str(g_iter) + " " + str(time) + " " + str(inception_score[0]) + "\n"
            #model.file.write(output)

            #real = model.real_images(images, model.number_of_images)
            generated = model.generate_img(z, model.number_of_images)
            #merged = torch.stack((real, generated), dim=1).view(-1,3,32,32)

            info = {
                'Wasserstein distance': Wasserstein_D.data,
                'Loss D': d_loss.data,
                'Loss G': g_cost.data,
                'Loss D Real': d_loss_real.data,
                'Loss D Fake': d_loss_fake.data,
                'images': wandb.Image(generated, caption="generated images"),
            }

            wandb.log(info)


    model.t_end = t.time()
    print('Time of training-{}'.format((model.t_end - model.t_begin)))
    #model.file.close()

    # Save the trained parameters
    model.save_model()


def evaluate(model, test_loader, config):
    model.load_model(os.path.join(config.ckpt_path,"discriminator.pkl"), os.path.join(config.ckpt_path,"generator.pkl"))
    z = model.get_torch_variable(torch.randn(config.batch_size, 100, 1, 1))
    samples = model.G(z)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.data.cpu()
    grid = make_grid(samples)
    print("Grid of 8x8 images saved to 'wgan_model_image.png'.")
    save_image(grid, os.path.join(config.img_output_dir, 'wgan_model_image.png'))
    


def main():

    configs = {
        "model":"wgan_gp",
        "channels":3,
        "batch_size": 64,
        "learning_rate":1e-4,
        "betas":(0.5, 0.999),
        "iters": 40000,
        "train": True,
        "cuda": True,
        "split": 1,
        "type":"train wgan"
"
    }

    wandb.init(project="547_term", config=configs)
    config = wandb.config
    config.ckpt_path = "ckpt/wgan_gp/split"+str(config.split)
    config.img_output_dir = "output/wgan_gp/split"+str(config.split)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WGAN_GP(config).to(device)
    
    # Prepare splitted dataset and loaders
    train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True)
    val_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True)
    train_transform = T.Compose([T.ToPILImage(), T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    val_transform = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    split_train_set = split_dataset(train_set, config.split, train_transform)
    split_val_test = split_dataset(val_set, config.split, val_transform)
    train_loader = DataLoader(split_train_set, batch_size=config.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(split_val_test, batch_size=config.batch_size, shuffle=False, num_workers=8)

    if config.train:
        train(model,train_loader, config)

    else:
        evaluate(model,val_loader, config)
        for i in range(50):
            model.generate_latent_walk(i)

if __name__ == "__main__":
    main()
