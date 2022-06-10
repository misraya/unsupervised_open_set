# TRAIN AE on BDD100k
from model.vanilla_ae_bdd import VanillaAE
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision
from model.self_sup_detector import get_self_supervised_detector
from model.vanilla_ae_bdd import VanillaAE
from data.bdddataset2 import OpenSetBDDDataset, myCollate, move_to_cuda
import wandb
wandb.init(project="open-set-project")


ae = VanillaAE(latent_size=128).cuda().train()
optimizer = torch.optim.Adam(ae.parameters())

batch_size=2
dataset = OpenSetBDDDataset(split=0, trainval='train', imgDir='../data/bdd/images', annDir='../data/bdd/labels/det_20')
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=myCollate)
for i in range(10):
    print("\n",f'  EPOCH:  {str(i)}', "\n")
    for (img_numpys, img_tensors, annss) in tqdm(train_loader):
        img_tensors, annss = move_to_cuda([img_tensors, annss])

        img_tensors_batch = torch.stack(img_tensors)
        
        # normalize
        means = img_tensors_batch.mean(dim=(0,2,3))
        stds = img_tensors_batch.std(dim=(0,2,3))
        img_tensors_batch = torchvision.transforms.functional.normalize(img_tensors_batch, means, stds)
        
        regen_batch = ae(img_tensors_batch)
        loss = torch.nn.functional.mse_loss(regen_batch, img_tensors_batch)
        loss.backward()
        optimizer.step()
        wandb.log({'reconstruction mse loss': loss.item()})
        
    torch.save(ae, f'ckpts/ae_n_bdd_epoch_{str(i)}.pth')
