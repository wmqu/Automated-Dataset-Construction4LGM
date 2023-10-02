from os.path import splitext
from os import listdir
from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree
from glob import glob
import numpy as np
import scipy.io as scio

from ..dataloader import umd_dataset_utils


class UMDDataSet(Dataset):
    """Read and parse UMD datasets"""
    def __init__(self, umd_root, transforms=None, file_name: str = "train"):
        self.root = umd_root + '\\' + file_name
        self.img_root = os.path.join(self.root, "rgb/")
        self.mask_root = os.path.join(self.root, "mask/")
        self.rgb_suffix = '_rgb'
        self.masks_suffix = '_label'
        # Loading images.
        self.rgb_ids = [splitext(file)[0] for file in listdir(self.img_root) if not file.startswith('.')]
        self.masks_ids = [splitext(file)[0] for file in listdir(self.mask_root) if not file.startswith('.')]
        assert (len(self.rgb_ids) == len(self.masks_ids))
        print(f'Dataset has {len(self.rgb_ids)} examples .. {umd_root}')

        # sorting images.
        self.rgb_ids = np.sort(np.array(self.rgb_ids))
        self.masks_ids = np.sort(np.array(self.masks_ids))
        self.transforms = transforms

    def __len__(self):
        return len(self.rgb_ids)

    def __getitem__(self, index):
        # loading images.
        idx_rgb = self.rgb_ids[index]
        idx_split = idx_rgb.strip().split('_')
        idx = idx_split[0] + "_" + idx_split[1] + "_" + idx_split[2]
        img_file = glob(self.img_root + idx + self.rgb_suffix + '.*')
        mask_file = glob(self.mask_root + idx + self.masks_suffix + '.*')

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {idx}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'

        image = Image.open(img_file[0]).convert('RGB')
        H, W = image.size[0], image.size[1]
        data = scio.loadmat(mask_file[0])
        mask = data.get("gt_label")

        obj_name = idx.split("_")[0]
        obj_id = umd_dataset_utils.map_obj_name_to_id(obj_name)

        # Get obj bbox from affordance mask.
        foreground_mask = np.ma.getmaskarray(np.ma.masked_not_equal(mask, 0)).astype(np.uint8)
        obj_boxes = umd_dataset_utils.get_bbox(mask=foreground_mask, obj_ids=np.array([1]), img_width=H, img_height=W)
        area = (obj_boxes[:, 3] - obj_boxes[:, 1]) * (obj_boxes[:, 2] - obj_boxes[:, 0])
        iscrowd = []
        iscrowd.append(0)

        obj_ids = np.array([obj_id])
        obj_boxes = np.array(obj_boxes)
        iscrowd = np.array(iscrowd)

        # convert everything into a torch.Tensor
        target = {}
        target["image_id"] = torch.tensor([index])
        target["boxes"] = torch.as_tensor(obj_boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(obj_ids, dtype=torch.int64)
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)
        target["area"] = torch.as_tensor(area, dtype=torch.float32)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def coco_index(self, index):

        idx_rgb = self.rgb_ids[index]
        idx_split = idx_rgb.strip().split('_')
        idx = idx_split[0] + "_" + idx_split[1] + "_" + idx_split[2]
        img_file = glob(self.img_root + idx + self.rgb_suffix + '.*')
        mask_file = glob(self.mask_root + idx + self.masks_suffix + '.*')

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {idx}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'

        image = Image.open(img_file[0]).convert('RGB')
        data_height, data_width = image.size[0], image.size[1]
        data = scio.loadmat(mask_file[0])
        mask = data.get("gt_label")

        obj_name = idx.split("_")[0]
        obj_id = umd_dataset_utils.map_obj_name_to_id(obj_name)

        # Get obj bbox from affordance mask.
        foreground_mask = np.ma.getmaskarray(np.ma.masked_not_equal(mask, 0)).astype(np.uint8)
        obj_boxes = umd_dataset_utils.get_bbox(mask=foreground_mask, obj_ids=np.array([1]), img_width=data_height, img_height=data_width)
        area = (obj_boxes[:, 3] - obj_boxes[:, 1]) * (obj_boxes[:, 2] - obj_boxes[:, 0])

        obj_ids = np.array([obj_id])
        obj_boxes = np.array(obj_boxes)
        iscrowd = []
        iscrowd.append(0)
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(obj_boxes, dtype=torch.float32)
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([index])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

