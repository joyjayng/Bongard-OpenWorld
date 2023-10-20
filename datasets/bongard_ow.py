import json
import logging
import os
import pickle

import cv2
import numpy as np
import torch
from detectron2.data import transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.dataset import register

logger = logging.getLogger(__name__)


@register('bongard-ow')
class BongardOWDataset(Dataset):

    def __init__(self,
                 im_root,
                 problem_file,
                 bbox_file,
                 resize_224=True,
                 use_clip_stat=False,
                 basic_augmentation=True,
                 extra_augmentation=False,
                 image_size=256,
                 max_image_size=256*2):
        self.im_root = im_root
        self.problems = json.load(open(problem_file, 'r'))
        self.boxes = pickle.load(open(bbox_file, 'rb'))

        self.image_transformations = [T.NoOpTransform()]
        if resize_224:
            self.image_transformations.append(
                T.Resize(224),
            )
        if basic_augmentation:
            self.image_transformations.append(
                T.ResizeShortestEdge(image_size, max_image_size, 'range')
            )
            self.image_transformations.append(
                T.RandomFlip(horizontal=True, vertical=False)
            )
        if extra_augmentation:
            self.image_transformations.append(
                T.RandomBrightness(0.6, 1.4)
            )
            self.image_transformations.append(
                T.RandomContrast(0.6, 1.4)
            )
            self.image_transformations.append(
                T.RandomSaturation(0.6, 1.4)
            )
        self.image_transformations = T.AugmentationList(
            self.image_transformations
        )

        if use_clip_stat:
            # CLIP
            self.pix_mean = (0.48145466, 0.4578275, 0.40821073)
            self.pix_std = (0.26862954, 0.26130258, 0.27577711)
        else:
            # IN
            self.pix_mean = (0.485, 0.456, 0.406)
            self.pix_std = (0.229, 0.224, 0.225)

    def get_image(self, path, boxes):
        im = cv2.imread(os.path.join(self.im_root, path)).astype(np.float32)
        aug_input = T.AugInput(im, boxes=boxes)
        self.image_transformations(aug_input)
        im = aug_input.image
        boxes = aug_input.boxes
        for i in range(3):
            im[:, :, i] = (im[:, :, i] / 255. - self.pix_mean[i]) / self.pix_std[i]
        # H,W,C -> C,H,W
        return (
            torch.as_tensor(np.ascontiguousarray(im.transpose(2, 0, 1))),
            torch.as_tensor(np.ascontiguousarray(boxes))
        )

    def _pad_tensor(self, tensor_list):
        max_imh, max_imw = -1, -1
        for tensor_i in tensor_list:
            imh, imw = tensor_i.shape[-2], tensor_i.shape[-1]
            max_imh = max(max_imh, imh)
            max_imw = max(max_imw, imw)

        for idx, tensor_i in enumerate(tensor_list):
            pad_tensor_i = tensor_i.new_full(list(tensor_i.shape[:-2]) + [max_imh, max_imw], 0)
            imh, imw = tensor_i.shape[-2], tensor_i.shape[-1]
            pad_tensor_i[..., :imh, :imw].copy_(tensor_i)
            tensor_list[idx] = pad_tensor_i
        return tensor_list

    def _norm_boxes(self, image_size, boxes):
        boxes[:, 0] /= image_size[-1]
        boxes[:, 1] /= image_size[-2]
        boxes[:, 2] /= image_size[-1]
        boxes[:, 3] /= image_size[-2]
        return boxes

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, index):
        problem = self.problems[index]
        concept = problem['concept']
        caption = problem['caption']
        all_img, all_boxes, all_norm_boxes = [], [], []
        for path in problem['imageFiles']:
            img, boxes = self.get_image(
                path,
                list(self.boxes.values())[0]['boxes']
                # self.boxes[path.split('/')[-1]]['boxes']
            )
            all_img.append(img)
            all_boxes.append(boxes)
            all_norm_boxes.append(self._norm_boxes(img.shape, boxes))
        # pad to the right bottom, so no need to modify boxes
        all_img = self._pad_tensor(all_img)
        all_boxes = self._pad_tensor(all_boxes)
        all_norm_boxes = self._pad_tensor(all_norm_boxes)

        shot_ims = torch.stack(all_img[:6] + all_img[7:-1], dim=0).float()
        shot_boxes = torch.stack(all_boxes[:6] + all_boxes[7:-1], dim=0).float()
        shot_norm_boxes = torch.stack(all_norm_boxes[:6] + all_norm_boxes[7:-1], dim=0).float()
        shot_labels = torch.cat([torch.zeros(6), torch.ones(6)]).long()

        query_ims = torch.stack([all_img[6], all_img[-1]], dim=0).float()
        query_boxes = torch.stack([all_boxes[6], all_boxes[-1]], dim=0).float()
        query_norm_boxes = torch.stack([all_norm_boxes[6], all_norm_boxes[-1]], dim=0).float()
        query_labels = torch.Tensor([0, 1]).long()

        return {
            'uid': problem['uid'],
            'concept': concept,
            'caption': caption,
            'commonSense': problem['commonSense'],
            'shot_ims': shot_ims,
            'shot_boxes': shot_boxes,
            'shot_norm_boxes': shot_norm_boxes,
            'shot_labels': shot_labels,
            'query_ims': query_ims,
            'query_boxes': query_boxes,
            'query_norm_boxes': query_norm_boxes,
            'query_labels': query_labels,
        }


def collate_images_boxes_dict(batch):
    def _pad_tensor(tensor_list):
        max_imh, max_imw = -1, -1
        for tensor_i in tensor_list:
            imh, imw = tensor_i.shape[-2], tensor_i.shape[-1]
            max_imh = max(max_imh, imh)
            max_imw = max(max_imw, imw)

        for idx, tensor_i in enumerate(tensor_list):
            pad_tensor_i = tensor_i.new_full(list(tensor_i.shape[:-2]) + [max_imh, max_imw], 0)
            imh, imw = tensor_i.shape[-2], tensor_i.shape[-1]
            pad_tensor_i[..., :imh, :imw].copy_(tensor_i)
            tensor_list[idx] = pad_tensor_i
        return tensor_list

    keys = list(batch[0].keys())
    batched_dict = {}
    for k in keys:
        data_list = []
        for batch_i in batch:
            data_list.append(batch_i[k])
        if isinstance(data_list[0], torch.Tensor):
            if len(data_list[0].shape) > 1:
                data_list = _pad_tensor(data_list)
            data_list = torch.stack(data_list, dim=0)
        batched_dict[k] = data_list
    return batched_dict
