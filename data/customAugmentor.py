import torchvision.transforms as T
import albumentations as A
import numpy as np
import torch

class CustomAugmentor():
    def __init__(self):
        

        a0 = A.Compose([A.VerticalFlip(p=1), 
                        A.HorizontalFlip(p=1), 
                        A.Affine(rotate=90, fit_output=True, p=1)], 
                       bbox_params=A.BboxParams(format='pascal_voc', 
                                                label_fields=['class_labels'])) # factor 1 
        a1 = A.Compose([A.VerticalFlip(p=1), 
                        A.HorizontalFlip(p=1), 
                        A.Affine(rotate=180, fit_output=True, p=1)], 
                       bbox_params=A.BboxParams(format='pascal_voc', 
                                                label_fields=['class_labels'])) # factor 2
        a2 = A.Compose([A.VerticalFlip(p=1), 
                        A.HorizontalFlip(p=1), 
                        A.Affine(rotate=270, fit_output=True, p=1)], 
                       bbox_params=A.BboxParams(format='pascal_voc', 
                                                label_fields=['class_labels'])) # factor 3
        a3 = A.Compose([A.VerticalFlip(p=1), 
                        A.HorizontalFlip(p=1)], 
                       bbox_params=A.BboxParams(format='pascal_voc', 
                                                label_fields=['class_labels'])) # factor 0
        a4 = A.Compose([A.VerticalFlip(p=1), 
                        A.Affine(rotate=90, fit_output=True, p=1)], 
                       bbox_params=A.BboxParams(format='pascal_voc', 
                                                label_fields=['class_labels'])) # factor 1
        a5 = A.Compose([A.VerticalFlip(p=1), 
                        A.Affine(rotate=180, fit_output=True, p=1)], 
                       bbox_params=A.BboxParams(format='pascal_voc', 
                                                label_fields=['class_labels'])) # factor 2
        a6 = A.Compose([A.VerticalFlip(p=1), 
                        A.Affine(rotate=270, fit_output=True, p=1)], 
                       bbox_params=A.BboxParams(format='pascal_voc', 
                                                label_fields=['class_labels'])) # factor 3
        a7 = A.Compose([A.VerticalFlip(p=1)], 
                       bbox_params=A.BboxParams(format='pascal_voc', 
                                                label_fields=['class_labels'])) # factor 0
        a_ = A.Compose([A.Affine(rotate=90, fit_output=True, p=1)], 
                       bbox_params=A.BboxParams(format='pascal_voc', 
                                                label_fields=['class_labels'])) # test



        t0 = T.Compose([T.RandomVerticalFlip(p=1.), 
                        T.RandomHorizontalFlip(p=1.), 
                        T.RandomRotation(degrees=[270,270], 
                                         expand=True)]) # factor 1
        t1 = T.Compose([T.RandomVerticalFlip(p=1.), 
                        T.RandomHorizontalFlip(p=1.), 
                        T.RandomRotation(degrees=[180,180], 
                                         expand=True)]) # factor 2
        t2 = T.Compose([T.RandomVerticalFlip(p=1.), 
                        T.RandomHorizontalFlip(p=1.), 
                        T.RandomRotation(degrees=[90,90], 
                                         expand=True)]) # factor 3
        t3 = T.Compose([T.RandomVerticalFlip(p=1.), 
                        T.RandomHorizontalFlip(p=1.)]) # factor 0
        t4 = T.Compose([T.RandomVerticalFlip(p=1.), 
                        T.RandomRotation(degrees=[270,270], 
                                         expand=True)]) # factor 1
        t5 = T.Compose([T.RandomVerticalFlip(p=1.), 
                        T.RandomRotation(degrees=[180,180], 
                                         expand=True)]) # factor 2
        t6 = T.Compose([T.RandomVerticalFlip(p=1.), 
                        T.RandomRotation(degrees=[90,90], 
                                         expand=True)]) # factor 3
        t7 = T.Compose([T.RandomVerticalFlip(p=1.)]) # factor 0
        t_ = T.Compose([T.RandomRotation(degrees=[270,270], 
                                         expand=True)])

        #self.aug_factors = {0: 1, 1: 2, 2: 3, 3: 0, 4: 1, 5: 2, 6: 3, 7: 0}
        self.at = [a0, a1, a2, a3, a4, a5, a6, a7, a_]
        self.tt = [t0, t1, t2, t3, t4, t5, t6, t7, t_]
        
    def augment_annotations(self, img_numpys, annss, trans, index,):
        new_img_numpys, new_anns = [], []
        for img_numpy, ann in zip(img_numpys, annss):
            transformed = trans(image=img_numpy, 
                                bboxes=ann['boxes'].clone().cpu().numpy(),
                                class_labels=ann['labels'].clone().cpu().numpy())
            new_img_numpys.append(transformed['image'])
            bboxes = torch.Tensor([list(bbox) for bbox in transformed['bboxes']]).cuda()
            labels = torch.Tensor(transformed['class_labels']).long().cuda()

            new_anns.append({'boxes': bboxes,
                             'labels': labels,
                             'aug_labels': index*ann['aug_labels'].clone()})

        return new_img_numpys, new_anns
    
    def augment(self, img_numpys, img_tensors, reg_tensors, annss):
        aug_i = np.random.randint(8) # select random [0,7] augmenttaion labels(also the index)
        #aug_factor = self.aug_factors[aug_i] # find corresponding rotation factor for albumentations
        #print(f'aug i/label: {aug_i}, aug factor: {aug_factor}')

        t_i = self.tt[aug_i] # get appropriate torchvision transform function
        a_i = self.at[aug_i] # get appropriate albumentation transform function

        img_tensors_aug = [t_i(img_tensor.clone()) for img_tensor in img_tensors] # transform all img_tensors in list-batch
        reg_tensors_aug = [t_i(reg_tensor.clone()) for reg_tensor in reg_tensors] # transform all regenerated img_tensors in list-batch
        img_numpys_aug, anns_aug = self.augment_annotations(img_numpys.copy(), annss.copy(), a_i, aug_i)
        
        return img_numpys_aug, img_tensors_aug, reg_tensors_aug, anns_aug
    
    def augment_without_reconstructions(self, img_numpys, img_tensors, annss):
        aug_i = np.random.randint(8) # select random [0,7] augmenttaion labels(also the index)
        #aug_factor = self.aug_factors[aug_i] # find corresponding rotation factor for albumentations
        #print(f'aug i/label: {aug_i}, aug factor: {aug_factor}')

        t_i = self.tt[aug_i] # get appropriate torchvision transform function
        a_i = self.at[aug_i] # get appropriate albumentation transform function

        img_tensors_aug = [t_i(img_tensor.clone()) for img_tensor in img_tensors] # transform all img_tensors in list-batchlist-batch
        img_numpys_aug, anns_aug = self.augment_annotations(img_numpys.copy(), annss.copy(), a_i, aug_i)
        
        return img_numpys_aug, img_tensors_aug, anns_aug
    
    
    
