# TRAIN DETECTOR on BDDD100k
from model.vanilla_ae_bdd import VanillaAE
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from model.self_sup_detector import get_self_supervised_detector
from model.vanilla_ae_bdd import VanillaAE
from data.bdddataset2 import OpenSetBDDDataset, myCollate, move_to_cuda
from data.customAugmentor import CustomAugmentor
import wandb
wandb.init(project="open-set-project")


detector = get_self_supervised_detector(True).cuda().train()
ae = torch.load('ae_bdd.pth').eval()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    detector = nn.DataParallel(detector)
    ae = nn.DataParallel(ae)
    batch_size = 2*torch.cuda.device_count()
else:
    batch_size = 2

dataset = OpenSetBDDDataset(split=0, trainval='train', imgDir='../data/bdd/images', annDir='../data/bdd/labels/det_20')
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, collate_fn=myCollate)


for p in ae.parameters():
    p.required_grad = False
optimizer_detector = torch.optim.Adam(detector.parameters())
alpha_1 = 0.8
alpha_2 = 0.2

augmentor = CustomAugmentor()

for epoch in range(5):
    for (img_numpys, img_tensors, annss) in tqdm(train_loader):
        img_tensors, annss = move_to_cuda([img_tensors, annss])
        
        # normalize
        img_tensors_batch = torch.stack(img_tensors) # convert list of 3D Tensors to one 4D Tensor
        means = img_tensors_batch.mean(dim=(0,2,3))
        stds = img_tensors_batch.std(dim=(0,2,3))
        img_tensors_batch = torchvision.transforms.functional.normalize(img_tensors_batch, means, stds) # normalized batch
        img_tensors =  [img_tensor.squeeze(0) for img_tensor in img_tensors_batch.chunk(batch_size,dim=0)] # convert to list of 3D tensors
        
        with torch.no_grad():
            regen = ae(img_tensors_batch).chunk(batch_size,dim=0)
            reg_tensors = [regen_tensor.squeeze(0) for regen_tensor in regen] # convert to list of 3D tensors

        img_numpys_aug, img_tensors_aug, reg_tensors_aug, anns_aug = augmentor.augment(img_numpys, img_tensors, reg_tensors, annss)

        orig_tensors = [torch.cat((img_tensor, reg_tensor)) for img_tensor, reg_tensor in zip(img_tensors, reg_tensors)]
        aug_tensors = [torch.cat((img_tensor, reg_tensor)) for img_tensor, reg_tensor in zip(img_tensors_aug, reg_tensors_aug)]

        result = detector(orig_tensors, annss)
        result_aug = detector(aug_tensors, anns_aug)

        # losses for open-set workflow
        l_clss = result['loss_classifier']
        l_ss = result_aug['loss_self_sup']
        openset_loss  = alpha_1*l_clss + alpha_2*l_ss

        # also need to train detector, check frcnn paper??:
        detector_loss = result['loss_box_reg'] + result['loss_objectness'] + result['loss_rpn_box_reg']

        # update for open-set losses
        optimizer_detector.zero_grad()
        openset_loss.backward(retain_graph=True)
        #optimizer_detector.step()

        # update for detector losses
        #optimizer_detector.zero_grad()
        detector_loss.backward()
        optimizer_detector.step()

        #losses.append(loss.item())
        wandb.log({'openset_loss': openset_loss.item(), 'l_clss': l_clss.item(), 'l_ss': l_ss.item(), 'detector_loss': detector_loss.item()})
    torch.save(detector, f'det_bdd_epoch_{str(epoch)}.pth')
