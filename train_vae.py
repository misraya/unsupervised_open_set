# TRAIN VAE on BDD100k
from model.vanilla_ae_bdd import VAE
from tqdm import tqdm
import torch
import torchvision
from torch.utils.data import DataLoader
from model.self_sup_detector import get_self_supervised_detector
from data.bdddataset2 import OpenSetBDDDataset, myCollate, move_to_cuda
import wandb
wandb.init(project="open-set-project")


vae = VAE(latent_size=128).cuda().train()
optimizer = torch.optim.Adam(vae.parameters())


dataset = OpenSetBDDDataset(split=0, trainval='train', imgDir='../data/bdd/images', annDir='../data/bdd/labels/det_20')
train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=8, pin_memory=True, collate_fn=myCollate)
for i in range(10):
    print("\n",f'  EPOCH:  {str(i)}', "\n")
    for (img_numpys, img_tensors, annss) in tqdm(train_loader):
        img_tensors, annss = move_to_cuda([img_tensors, annss])

        img_tensors_batch = torch.stack(img_tensors)
        
        # normalize
        means = img_tensors_batch.mean(dim=(0,2,3))
        stds = img_tensors_batch.std(dim=(0,2,3))
        img_tensors_batch = torchvision.transforms.functional.normalize(img_tensors_batch, means, stds)
        
        
        regen_batch, zus = vae(img_tensors_batch)
        
        u, s = zus.chunk(2, dim=1)
        KL_divergence = (-s - 0.5 + (torch.exp(2 * s) + u ** 2) * 0.5).mean(dim=0).sum()
        reconstruction_loss = torch.nn.functional.mse_loss(regen_batch, img_tensors_batch)
        elbo_loss = KL_divergence + reconstruction_loss
        elbo_loss.backward()
        optimizer.step()
        wandb.log({'elbo_loss': elbo_loss.item(), 'KL_divergence': KL_divergence.item(), 'reconstruction mse loss': reconstruction_loss.item()})
        
    torch.save(vae, f'ckpts/vae_n_bdd_epoch_{str(1)}.pth')
