import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torchvision.io import read_image
import cv2
import albumentations as A
import numpy as np

class OpenSetBDDDataset(Dataset):
    def __init__(self, split=0, trainval='train', imgDir='data/bdd/images', annDir='data/bdd/labels/det_20'):
        self.trainval = trainval # can be 'train' or 'val', determines how annotations are filtered
        
        self.imgDir = f'{imgDir}/{trainval}/'
        self.annDir = f'{annDir}/det_{trainval}_cocofmt.json'
        self.coco = COCO(self.annDir)

        self.ood_splits = [['bicycle', 'motorcycle', 'rider', 'train']]
        self.closed_splits = [['pedestrian', 'car', 'truck', 'bus', 'traffic light', 'traffic sign']]

        self.ood_categories = ['bicycle', 'motorcycle', 'rider', 'train'] # self.ood_splits[split] # [8, 7, 2, 6]
        self.closed_categories = ['pedestrian', 'car', 'truck', 'bus', 'traffic light', 'traffic sign'] # self.closed_splits[split] # [1, 3, 4, 5, 9, 10]
        self.full_categories = self.ood_categories + self.closed_categories
        self.remap_dict, self.unmap_dict, self.category_ids2names = self.remap_categories()

        self.aug_labels_dict = self.build_augmentation_labels_dict()
        self.imgIds = self.coco.getImgIds() # all images in dataset
        
        self.h_flip = A.Compose([A.HorizontalFlip(p=1.0)], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
        self.v_flip = A.Compose([A.VerticalFlip(p=1.0)], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
        self.rot_90 = A.Compose([A.augmentations.geometric.rotate.RandomRotate90()], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
        
    def numpy_2_ch_tensor(self, img_numpy):
        return torch.permute(torch.Tensor(img_numpy),(2,0,1))
        
    def remap_categories(self):
        # this is so that closed category ids such as [1, 3, 4, 5, 9, 10] get mapped to [0, 1, 2, 3, 4, 5]
        # then the detector's classification head only needs 6 output neurons, and loss can be computed easily
        remap_dict = {0: 0}
        unmap_dict = {0: 'background'}
        # To do: check what class 0 is supposed to mean for frcnn. background or...?
        for i, catId in enumerate(self.coco.getCatIds(catNms=self.closed_categories)):
            remap_dict[catId] = i + 1 #### IMPORTANT: save the 0 label for background
            unmap_dict[i+1] = self.coco.loadCats(ids=catId)[0]['name']
        
        remap_dict[catId+1] = i + 2 # add additional label for visualizations during inference
        unmap_dict[i+2] = "ood" # add additional label for visualizations during inference
        
        print(f'Remapping category labels... \n {remap_dict}')
        category_ids2names = dict(zip(remap_dict.keys(), unmap_dict.values()))
        return remap_dict, unmap_dict, category_ids2names
    
    def build_augmentation_labels_dict(self):
        aug_labels_dict = {'111': 0, # (v_flipped == 1 and h_flipped == 1 and factor == 1)
                           '003': 0, # (v_flipped == 0 and h_flipped == 0 and factor == 3)
                           '112': 1, # (v_flipped == 1 and h_flipped == 1 and factor == 2)
                           '000': 1, # (v_flipped == 0 and h_flipped == 0 and factor == 0)
                           '113': 2, # (v_flipped == 1 and h_flipped == 1 and factor == 3)
                           '001': 2, # (v_flipped == 0 and h_flipped == 0 and factor == 1)
                           '110': 3, # (v_flipped == 1 and h_flipped == 1 and factor == 0)
                           '002': 3, # (v_flipped == 0 and h_flipped == 0 and factor == 2)
                           '101': 4, # (v_flipped == 1 and h_flipped == 0 and factor == 1)
                           '013': 4, # (v_flipped == 0 and h_flipped == 1 and factor == 3)
                           '102': 5, # (v_flipped == 1 and h_flipped == 0 and factor == 2)
                           '010': 5, # (v_flipped == 0 and h_flipped == 1 and factor == 0)
                           '103': 6, # (v_flipped == 1 and h_flipped == 0 and factor == 3)
                           '011': 6, # (v_flipped == 0 and h_flipped == 1 and factor == 1)
                           '100': 7, # (v_flipped == 1 and h_flipped == 0 and factor == 0)
                           '012': 7  # (v_flipped == 0 and h_flipped == 1 and factor == 2)
                          }
        return aug_labels_dict
        
        
    def __len__(self):
        return len(self.imgIds)
    
    def __getitem__(self, idx):
        imgId = self.imgIds[idx]
        
        imgPath = f'{self.imgDir}/{self.coco.loadImgs([imgId])[0]["file_name"]}'
        #img_tensor = read_image(imgPath) # 3 x 720(H) x 1280(W)
        img_numpy = cv2.imread(imgPath) # 720(H) x 1280(W) x 3
        img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)/255
        #img_numpy = torch.permute( img_tensor.clone(), (1,2,0) ).numpy() # keep a copy in numpy formate for augmenting
        
        annIds = self.coco.getAnnIds(imgIds=imgId, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        anns = self.remove_ood_annotations(anns)
        anns = self.prepare_anns_frcnn_fmt(anns)
        
        #img_aug, anns_aug = self.augment(img_numpy.copy(), anns)
        
        img_tensor = self.numpy_2_ch_tensor(img_numpy)
        #img_aug_tensor = self.numpy_2_ch_tensor(img_aug)
        anns = self.convert_xywh_2_x1y1x2y2(anns)
        #anns_aug = self.convert_xywh_2_x1y1x2y2(anns_aug)
        return img_numpy, img_tensor, anns#, img_aug_tensor, anns_aug
    
    def convert_xywh_2_x1y1x2y2(self, anns):
        bboxes = anns['boxes']
        bboxes[:,2] = bboxes[:,2] + bboxes[:,0]
        bboxes[:,3] = bboxes[:,3] + bboxes[:,1]
        anns['boxes'] = bboxes
        return anns
    
    def remove_ood_annotations(self, anns):
        ood_ids = self.coco.getCatIds(catNms=self.ood_categories)
        new_anns = []
        for ann in anns:
            if ann['category_id'] in ood_ids:
                if self.trainval == 'val':
                    ann['category_id'] = 0 ## IMPORTANT: This sets the class as background so that detector can compute loss
                    new_anns.append(ann)
            else:
                new_anns.append(ann)
        return new_anns

    def prepare_anns_frcnn_fmt(self, anns):
        gt_bboxes = []
        gt_classes = []
        gt_augs = []
        for ann in anns:
            gt_bboxes.append(ann['bbox'])
            gt_classes.append(self.remap_dict[ann['category_id']])

        new_anns = {'boxes': torch.Tensor(gt_bboxes),
                    'labels': torch.Tensor(gt_classes).long(),
                    'aug_labels': torch.ones(len(gt_classes)).long() # the label 1 represents no augmentation via '112'
                   }
        return new_anns
    
    def augment(self, img_numpy, anns):
        bboxes = anns['boxes'].clone().numpy()
        labels = anns['labels'].clone().numpy()
        
        # apply horizontal_flip
        if np.random.rand() > 0.5:
            transformed = self.h_flip(image=img_numpy, 
                                      bboxes=bboxes, 
                                      class_labels=labels )
            img_numpy = transformed['image']
            bboxes = np.array([list(bbox) for bbox in transformed['bboxes']])
            labels = transformed['class_labels']
            
            h_flipped = 1
        else:
            h_flipped = 0
            
        # apply vertical_flip
        if np.random.rand() > 0.5:
            transformed = self.v_flip(image=img_numpy, 
                                      bboxes=bboxes, 
                                      class_labels=labels)
            img_numpy = transformed['image']
            bboxes = np.array([list(bbox) for bbox in transformed['bboxes']])
            labels = transformed['class_labels']
            
            v_flipped = 1
        else:
            v_flipped = 0
            
        # apply random rotate
        rot_factor = np.random.rand()
        if rot_factor < 0.25:
            factor = 0
        elif rot_factor < 0.5:
            factor = 1
        elif rot_factor < 0.75:
            factor = 2
        elif rot_factor < 1.0:
            factor = 3
            
        transformed = self.rot_90(image=img_numpy, 
                                  bboxes=bboxes, 
                                  class_labels=labels, 
                                  factor=factor)
        img_numpy = transformed['image']
        bboxes = np.array([list(bbox) for bbox in transformed['bboxes']])
        labels = transformed['class_labels']
        
        # determine augmentation label
        aug_key = str(v_flipped) + str(h_flipped) + str(factor)
        aug_label = self.aug_labels_dict[aug_key]
        
        
        # apply label to annotations for self-supervised branch
        new_anns = {'boxes': torch.Tensor(bboxes),
                    'labels': torch.Tensor(labels).long(),
                    'aug_labels': (aug_label*torch.ones(len(labels))).long()
                   }
        
        return img_numpy, new_anns
        
        
def myCollate(data):
    img_numpys, img_tensors, annss = [], [], []
    for sample in data:
        img_numpys.append(sample[0])
        img_tensors.append(sample[1])
        annss.append(sample[2])
    return img_numpys, img_tensors, annss

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
        
    
