from torchvision.transforms.functional import normalize
import torch.nn as nn
import numpy as np
import os
from PIL import Image

def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)

class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def getImageMatrix(image_path):
    img = Image.open(image_path)
    w = img.width  # 图片的宽
    h = img.height  # 图片的高
    np.set_printoptions(threshold=1e7)    # 设置阈值，不用...显示
    return np.zeros((h, w), dtype=np.uint8)


def iit_map_obj_id_to_aff_id(obj_id):
    aff_ids = []
    if obj_id == 0:  # "bowl"
        aff_ids.append([1, 9])
    elif obj_id == 1:  # "tvm"
        aff_ids.append([3])
    elif obj_id == 2:  # "pan"
        aff_ids.append([1, 5])
    elif obj_id == 3:  # "hammer"
        aff_ids.append([7, 5])
    elif obj_id == 4:  # "knife"
        aff_ids.append([2, 5])
    elif obj_id == 5:  # "cup"
        aff_ids.append([1, 9])
    elif obj_id == 6:  # "drill"
        aff_ids.append([4, 5])
    elif obj_id == 7:  # "racket"
        aff_ids.append([6, 5])
    elif obj_id == 8:  # "spatula"
        aff_ids.append([8, 5])
    elif obj_id == 9:  # "bottle"
        aff_ids.append([1, 5])
    else:
        assert (" --- Object does not exist in IIT dataset --- ")
    return aff_ids
def iit_map_obj_name_to_id(obj_name):

    if obj_name == "bowl":
        return 0
    elif obj_name == "tvm":
        return 1
    elif obj_name == "pan":
        return 2
    elif obj_name == "hammer":
        return 3
    elif obj_name == "knife":
        return 4
    elif obj_name == "cup":
        return 5
    elif obj_name == "drill":
        return 6
    elif obj_name == "racket":
        return 7
    elif obj_name == "spatula":
        return 8
    elif obj_name == "bottle":
        return 9
    else:
        assert (" --- Object does not exist in IIT dataset --- ")
def umd_map_obj_id_to_aff_id(obj_id):
    aff_ids = []
    # for i in range(len(obj_ids)):
    #     obj_id = obj_ids[i]
    if obj_id == 0:
        aff_ids.append([0])
    elif obj_id == 1:  # "bowl"
        aff_ids.append([4])
    elif obj_id == 2:  # "cup"
        aff_ids.append([4, 7])
    elif obj_id == 3:  # "hammer"
        aff_ids.append([5, 1])
    elif obj_id == 4:  # "knife"
        aff_ids.append([2, 1])
    elif obj_id == 5:  # "ladle"
        aff_ids.append([4, 1])
    elif obj_id == 6:  # "mallet"
        aff_ids.append([5, 1])
    elif obj_id == 7:  # "mug"
        aff_ids.append([4, 1])
    elif obj_id == 8:  # "pot"
        aff_ids.append([4, 7])
    elif obj_id == 9:  # "saw"
        aff_ids.append([2, 1])
    elif obj_id == 10:  # "scissors"
        aff_ids.append([2, 1])
    elif obj_id == 11:  # "scoop"
        aff_ids.append([3, 1])
    elif obj_id == 12:  # "shears"
        aff_ids.append([2, 1])
    elif obj_id == 13:  # "shovel"
        aff_ids.append([3, 1])
    elif obj_id == 14:  # "spoon"
        aff_ids.append([3, 1])
    elif obj_id == 15:  # "tenderizer"
        aff_ids.append([5, 1])
    elif obj_id == 16:  # "trowel"
        aff_ids.append([3, 1])
    elif obj_id == 17:  # "turner"
        aff_ids.append([6, 1])
    else:
        assert (" --- Object does not exist in UMD dataloader --- ")
    return aff_ids
def umd_map_obj_name_to_id(obj_name):
    if obj_name == "bowl":
        return 1
    elif obj_name == "cup":
        return 2
    elif obj_name == "hammer":
        return 3
    elif obj_name == "knife":
        return 4
    elif obj_name == "ladle":
        return 5
    elif obj_name == "mallet":
        return 6
    elif obj_name == "mug":
        return 7
    elif obj_name == "pot":
        return 8
    elif obj_name == "saw":
        return 9
    elif obj_name == "scissors":
        return 10
    elif obj_name == "scoop":
        return 11
    elif obj_name == "shears":
        return 12
    elif obj_name == "shovel":
        return 13
    elif obj_name == "spoon":
        return 14
    elif obj_name == "tenderizer":
        return 15
    elif obj_name == "trowel":
        return 16
    elif obj_name == "turner":
        return 17
    else:
        assert (" --- Object does not exist in UMD dataloader --- ")