# -*- encoding:utf-8 -*-
from torch.utils.data import dataset
from tqdm import tqdm
import generate_mask.network as network
import generate_mask.utils as utils
import os
import random
import argparse
import numpy as np
import json
from torch.utils import data
from .datasets import IIT, UMD
from torchvision import transforms as T

from .get_affordance.iit_object_mask import get_object_mask
from .get_affordance.umd_object_mask import get_umd_object_mask
from .metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
from .utils import *


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options

    parser.add_argument("--input", type=str,
                        help="path to a cropped image directory",
                        default=r"")
    parser.add_argument("--dataset", type=str, default='umd',
                        choices=['iit', 'umd'], help='Name of training set')
    parser.add_argument("--rgb", type=str, default=r'',
                        help='image original path to get the original image size')
    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
                        choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to",
                        default=r"",
                        help="save single object segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default="", type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'umd':
        opts.num_classes = 8
        decode_fn = UMD.decode_target
    elif opts.dataset.lower() == 'iit':
        opts.num_classes = 10
        decode_fn = IIT.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    # loop through all files in a folder
    for root, dirs, files in os.walk(opts.input):
        for dir in dirs:
            path = os.path.join(root, dir)
            if os.path.isdir(path):
                # for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
                files = glob(os.path.join(path, '**/*.jpg'), recursive=True)
                if len(files) > 0:
                    image_files.extend(files)
    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.crop_val:
        transform = T.Compose([
            T.Resize(opts.crop_size),
            T.CenterCrop(opts.crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ]
        )
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)
    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            img_name = os.path.basename(img_path).split('.')[0]
            # Save the object mask in the same picture to a folder
            save_dir_name = img_path.split("\\")[-2]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0)  # To tensor of NCHW
            img = img.to(device)

            pred = model(img).max(1)[1].cpu().numpy()[0]  # HW
            if opts.dataset.lower() == 'iit':
                get_object_mask(img_name, pred, opts.rgb, save_dir_name, opts.save_val_results_to)
            elif opts.dataset.lower() == 'umd':
                get_umd_object_mask(img_name, pred, opts.rgb, save_dir_name, opts.save_val_results_to)




if __name__ == '__main__':
    main()
