from models.self_sup_detector import get_self_supervised_detector
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets.bdddataset import OpenSetBDDDataset, myCollate

def move_to_cuda(main_list): #[img_tensors, annss, img_tensor_augs, anns_augs]
    new_main_list = []
    for batch_item in main_list:
        if type(batch_item[0]) == dict:
            new_batch_item = []
            for dict_item in batch_item:
                new_dict_item = {}
                for key, value in dict_item.items():
                    new_dict_item[key] = value.cuda()
                new_batch_item.append(new_dict_item)
            new_main_list.append(new_batch_item)
        elif type(batch_item[0]) == torch.Tensor:
            new_main_list.append([img_tensor.cuda() for img_tensor in batch_item])
    return new_main_list

if __name__=="__main__":
    model = get_self_supervised_detector(False)

    model.cuda()
    model.train()

    dataset = OpenSetBDDDataset(split=0, trainval='train', imgDir='data/bdd/images', annDir='data/bdd/labels/det_20')
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True, collate_fn=myCollate)

    #optimizer_ae = torch.optim.Adam(auto_encoder.parameters())
    optimizer_detector = torch.optim.Adam(model.parameters())
    alpha_1 = 0.8
    alpha_2 = 0.2
    losses = []
    for (img_tensors, annss, img_tensor_augs, anns_augs) in tqdm(train_loader):

        img_tensors, annss, img_tensor_augs, anns_augs = move_to_cuda([img_tensors, annss, img_tensor_augs, anns_augs])

        result = model(img_tensors, annss)
        result_aug = model(img_tensor_augs, anns_augs)

        # losses for open-set workflow
        l_clss = result['loss_classifier']
        l_ss = result_aug['loss_self_sup']
        loss  = alpha_1*l_clss + alpha_2*l_ss

        # also need to train detector, check frcnn paper??:
        #detector_loss = result['loss_box_reg'] + result['loss_objectness'] + result['loss_rpn_box_reg'] ### ????


        # update for open-set losses
        optimizer_detector.zero_grad()
        loss.backward()
        optimizer_detector.step()

        # update for detector losses
        #optimizer_detector.zero_grad()
        #detector_loss.backward()
        #optimizer_detector.step()

        losses.append(loss.item())

        break