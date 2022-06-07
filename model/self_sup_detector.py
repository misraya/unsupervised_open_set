import mmcv
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection.faster_rcnn import RoIHeads
from torchvision.models.detection.faster_rcnn import *
from torchvision.models.detection.roi_heads import *

"""replacement  Roi head, this inherits FastRCNNPredictor and contains the deafult bbox head and classification head, and then we added a head to predict the augmenttaion"""
class FastRCNNPredictorPlus(FastRCNNPredictor):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.
    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels, num_classes)
        self.aug_score = nn.Linear(in_channels, 8)

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        aug_scores = self.aug_score(x)

        return scores, bbox_deltas, aug_scores
    
    
def self_supervised_loss(aug_logits, aug_labels):
    aug_labels = torch.cat(aug_labels, dim=0)
    return F.cross_entropy(aug_logits, aug_labels)

"""replacement for model.roi_heads(RoIHeads), this one modifies the forward and other helper functions to include a computation for the loss from the augmentation prediction head(self supervised part)"""
class RoIHeadsPlus(RoIHeads):
    def __init__(self, 
                 box_roi_pool,
                 box_head,
                 box_predictor,
                 box_fg_iou_thresh,
                 box_bg_iou_thresh,
                 box_batch_size_per_image,
                 box_positive_fraction,
                 bbox_reg_weights,
                 box_score_thresh,
                 box_nms_thresh,
                 box_detections_per_img):
        super().__init__(box_roi_pool,
                         box_head,
                         box_predictor,
                         box_fg_iou_thresh,
                         box_bg_iou_thresh,
                         box_batch_size_per_image,
                         box_positive_fraction,
                         bbox_reg_weights,
                         box_score_thresh,
                         box_nms_thresh,
                         box_detections_per_img)
    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError("target labels must of int64 type, instead got {t['labels'].dtype}")
                if not t["aug_labels"].dtype == torch.int64:  ########
                    raise TypeError("target aug labels must of int64 type, instead got {t['aug_labels'].dtype}") ########
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

        if self.training:
            proposals, matched_idxs, labels, regression_targets, aug_labels = self.select_training_samples(proposals, targets)
            #aug_labels = [t["aug_labels"] for t in targets]
        else:
            labels = None
            regression_targets = None
            matched_idxs = None
            aug_labels = None ######################

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression, aug_logits = self.box_predictor(box_features) #################
#         print(targets[0]['labels'].shape)
#         print(labels[0].shape)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            loss_self_sup = self_supervised_loss(aug_logits, aug_labels) ############
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg, "loss_self_sup": loss_self_sup} ################ 
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                if matched_idxs is None:
                    raise ValueError("if in trainning, matched_idxs should not be None")

                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                if targets is None or pos_matched_idxs is None or mask_logits is None:
                    raise ValueError("targets, pos_matched_idxs, mask_logits cannot be None when training")

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {"loss_mask": rcnn_loss_mask}
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if (
            self.keypoint_roi_pool is not None
            and self.keypoint_head is not None
            and self.keypoint_predictor is not None
        ):
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                if matched_idxs is None:
                    raise ValueError("if in trainning, matched_idxs should not be None")

                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                if targets is None or pos_matched_idxs is None:
                    raise ValueError("both targets and pos_matched_idxs should not be None when in training mode")

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
                )
                loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
            else:
                if keypoint_logits is None or keypoint_proposals is None:
                    raise ValueError(
                        "both keypoint_logits and keypoint_proposals should not be None when not in training mode"
                    )

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps
            losses.update(loss_keypoint)

        return result, losses
    
    def select_training_samples(
        self,
        proposals,  # type: List[Tensor]
        targets,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        gt_augs = [t["aug_labels"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
#         print(gt_boxes[0].shape)
#         print(gt_labels[0].shape)
#         print(proposals[0].shape)        
        
        matched_idxs, labels, augs = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels, gt_augs)
#         print('matched idx',matched_idxs[0].shape)
#         print(labels[0].shape)
        
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            augs[img_id] = augs[img_id][img_sampled_inds] ####################
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
#         print(labels[0].shape)
        return proposals, matched_idxs, labels, regression_targets, augs
    
    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels, gt_augs):
            # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
            matched_idxs = []
            labels = []
            augs = []
            for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, gt_augs_in_image in zip(proposals, gt_boxes, gt_labels, gt_augs):

                if gt_boxes_in_image.numel() == 0:
                    # Background image
                    device = proposals_in_image.device
                    clamped_matched_idxs_in_image = torch.zeros(
                        (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                    )
                    labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
                    augs_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device) ####
                else:
                    #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                    match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                    matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                    clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                    labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                    labels_in_image = labels_in_image.to(dtype=torch.int64)

                    augs_in_image = gt_augs_in_image[clamped_matched_idxs_in_image] ############3
                    augs_in_image = augs_in_image.to(dtype=torch.int64) ################

                    # Label background (below the low threshold)
                    bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                    labels_in_image[bg_inds] = 0

                    # Label ignore proposals (between low and high thresholds)
                    ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                    labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

                matched_idxs.append(clamped_matched_idxs_in_image)
                labels.append(labels_in_image)
                augs.append(augs_in_image)
            return matched_idxs, labels, augs

# if __name__ = "__main__":
   
def get_self_supervised_detector(replace_first_conv=True):
# create a pretrained, default fatsrercnn with resnet50 backbone and fpn
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    if replace_first_conv:
        # replace the first 3-channel conv with a 6-channel conv
        # clone the weights
        # replace first conv layer
        conv1 = torch.nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            conv1.weight[:,:3,:,:] = model.backbone.body.conv1.weight.clone()
            conv1.weight[:,3:,:,:] = model.backbone.body.conv1.weight.clone()
            model.backbone.body.conv1 = conv1
            
            model.transform.image_mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
            model.transform.image_std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]

    # create new instance of FastRCNNPredictorPlus with same args as model.roi_heads.box_predictor(FastRCNNPredictor)
    # clone the weights from the pretrained model
    # repace model.roi_heads.box_predictor
    in_channels, out_classes = model.roi_heads.box_predictor.cls_score.in_features, model.roi_heads.box_predictor.cls_score.out_features
    
    # TO DO: FIX load_state_dict() MISMATCH WHEN USING num_classes = 7!!!! Need to skip specific key
    box_predictor = FastRCNNPredictorPlus(in_channels=in_channels, num_classes=7) # out_classes# 7 # <---------------- IMPORTANT!! NEVERMIND, it's a single linear layer, so you don't need to transfer any hidden layer weights
#     box_predictor.cls_score.load_state_dict(model.roi_heads.box_predictor.cls_score.state_dict(), strict=False)
#     box_predictor.bbox_pred.load_state_dict(model.roi_heads.box_predictor.bbox_pred.state_dict(), strict=False)
    model.roi_heads.box_predictor = box_predictor



    # create new instance of RoIHeadsPlus with default/actual args
    # clone weights if necessary
    # replace model.roi_heads
    roi_heads = RoIHeadsPlus(model.roi_heads.box_roi_pool,
                             model.roi_heads.box_head,
                             model.roi_heads.box_predictor,
                             0.5,
                             0.5,
                             512,
                             0.25,
                             None,
                             0.05,
                             0.5,
                             100)
    roi_heads.load_state_dict(model.roi_heads.state_dict())
    model.roi_heads = roi_heads
    
    
    return model
