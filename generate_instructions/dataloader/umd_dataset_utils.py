import cv2
import numpy as np
import torch


# Class distributions.
OBJ_IDS_DISTRIBUTION = np.array([1000, 48, 62, 32, 131, 55, 33, 183, 16, 45, 90, 16, 24, 14, 77, 28, 44, 102]) / 1000
# Affordance distributions.
AFF_IDS_DISTRIBUTION = np.array([813, 289, 137, 355, 93, 116, 261]) / 2064
# num objects
NUM_OBJECT_CLASSES = 18

def get_bbox(mask, obj_ids, img_width, img_height, _step=40):
    border_list = np.arange(start=0, stop=np.max([img_width, img_height]) + _step, step=_step)
    # init boxes.
    boxes = np.zeros([len(obj_ids), 4], dtype=np.int32)
    for idx, obj_id in enumerate(obj_ids):
        rows = np.any(mask == obj_id, axis=1)
        cols = np.any(mask == obj_id, axis=0)

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        y2 += 1
        x2 += 1
        r_b = y2 - y1
        for tt in range(len(border_list)):
            if r_b > border_list[tt] and r_b < border_list[tt + 1]:
                r_b = border_list[tt + 1]
                break
        c_b = x2 - x1
        for tt in range(len(border_list)):
            if c_b > border_list[tt] and c_b < border_list[tt + 1]:
                c_b = border_list[tt + 1]
                break
        center = [int((y1 + y2) / 2), int((x1 + x2) / 2)]
        y1 = center[0] - int(r_b / 2)
        y2 = center[0] + int(r_b / 2)
        x1 = center[1] - int(c_b / 2)
        x2 = center[1] + int(c_b / 2)
        if y1 < 0:
            delt = -y1
            y1 = 0
            y2 += delt
        if x1 < 0:
            delt = -x1
            x1 = 0
            x2 += delt
        if y2 > img_width:
            delt = y2 - img_width
            y2 = img_width
            y1 -= delt
        if x2 > img_height:
            delt = x2 - img_height
            x2 = img_height
            x1 -= delt
        # x1,y1 ------
        # |          |
        # |          |
        # |          |
        # --------x2,y2
        # cv2.rectangle(img_bbox, (x1, y1), (x2, y2), (255, 0, 0), 2)
        boxes[idx] = np.array([x1, y1, x2, y2])
    return boxes
def map_obj_name_to_id(obj_name):
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

def map_obj_id_to_name(obj_id):
    if obj_id == 1:
        return "bowl"
    elif obj_id == 2:
        return "cup"
    elif obj_id == 3:
        return "hammer"
    elif obj_id == 4:
        return "knife"
    elif obj_id == 5:
        return "ladle"
    elif obj_id == 6:
        return "mallet"
    elif obj_id == 7:
        return "mug"
    elif obj_id == 8:
        return "pot"
    elif obj_id == 9:
        return "saw"
    elif obj_id == 10:
        return "scissors"
    elif obj_id == 11:
        return "scoop"
    elif obj_id == 12:
        return "shears"
    elif obj_id == 13:
        return "shovel"
    elif obj_id == 14:
        return "spoon"
    elif obj_id == 15:
        return "tenderizer"
    elif obj_id == 16:
        return "trowel"
    elif obj_id == 17:
        return "turner"
    else:
        assert (" --- Object does not exist in UMD dataloader --- ")

def map_aff_id_to_name(aff_id):
    if aff_id == 1:
        return "grasp"
    elif aff_id == 2:
        return "cut"
    elif aff_id == 3:
        return "scoop"
    elif aff_id == 4:
        return "contain"
    elif aff_id == 5:
        return "pound"
    elif aff_id == 6:
        return "support"
    elif aff_id == 7:
        return "wrap-grasp"
    else:
        assert (" --- Affordance does not exist in UMD dataloader --- ")
def map_name_to_aff_id(name):
    if name == "grasp":
        return 1
    elif name == "cut":
        return 2
    elif name == "scoop":
        return 3
    elif name == "contain":
        return 4
    elif name == "pound":
        return 5
    elif name == "support":
        return 6
    elif name == "wrap-grasp":
        return 7
    else:
        assert (" --- Affordance does not exist in UMD dataloader --- ")

def map_obj_id_to_aff_id(obj_id):
    aff_ids = []
    # for i in range(len(obj_ids)):
    #     obj_id = obj_ids[i]
    if obj_id == 0:  # "bowl"
        aff_ids.append([0])
    elif obj_id == 1:  # "bowl"
        aff_ids.append([4])
    elif obj_id == 2:  # "cup"
        aff_ids.append([4, 7])
    elif obj_id == 3:  # "hammer"
        aff_ids.append([1, 5])
    elif obj_id == 4:  # "knife"
        aff_ids.append([1, 2])
    elif obj_id == 5:  # "ladle"
        aff_ids.append([1, 4])
    elif obj_id == 6:  # "mallet"
        aff_ids.append([1, 5])
    elif obj_id == 7:  # "mug"
        aff_ids.append([1, 4, 7])
    elif obj_id == 8:  # "pot"
        aff_ids.append([4, 7])
    elif obj_id == 9:  # "saw"
        aff_ids.append([1, 2])
    elif obj_id == 10:  # "scissors"
        aff_ids.append([1, 2])
    elif obj_id == 11:  # "scoop"
        aff_ids.append([1, 3])
    elif obj_id == 12:  # "shears"
        aff_ids.append([1, 2])
    elif obj_id == 13:  # "shovel"
        aff_ids.append([1, 6])
    elif obj_id == 14:  # "spoon"
        aff_ids.append([1, 3])
    elif obj_id == 15:  # "tenderizer"
        aff_ids.append([1, 5])
    elif obj_id == 16:  # "trowel"
        aff_ids.append([1, 3])
    elif obj_id == 17:  # "turner"
        aff_ids.append([1, 6])
    else:
        assert (" --- Object does not exist in UMD dataloader --- ")
    return aff_ids

def format_obj_ids_to_aff_ids_list(obj_ids, aff_ids):
    if len(obj_ids) == 0:
        return []
    else:
        _aff_ids_list = []
        for i in range(len(obj_ids)):
            _aff_ids_list.append(list(aff_ids))
        return _aff_ids_list

def colorize_bbox(obj_id):

    increment = 255 / NUM_OBJECT_CLASSES

    if obj_id == "bowl":
        return 1 * increment
    elif obj_id == "cup":
        return 2 * increment
    elif obj_id == "hammer":
        return 3 * increment
    elif obj_id == "knife":
        return 4 * increment
    elif obj_id == "ladle":
        return 5 * increment
    elif obj_id == "mallet":
        return 6 * increment
    elif obj_id == "mug":
        return 7 * increment
    elif obj_id == "pot":
        return 8 * increment
    elif obj_id == "saw":
        return 9 * increment
    elif obj_id == "scissors":
        return 10 * increment
    elif obj_id == "scoop":
        return 11 * increment
    elif obj_id == "shears":
        return 12 * increment
    elif obj_id == "shovel":
        return 13 * increment
    elif obj_id == "spoon":
        return 14 * increment
    elif obj_id == "tenderizer":
        return 15 * increment
    elif obj_id == "trowel":
        return 16 * increment
    elif obj_id == "turner":
        return 17 * increment
    else:
        assert (" --- Object does not exist in UMD dataloader --- ")

def format_target_data(image, target):
    height, width = image.shape[0], image.shape[1]

    # original mask and binary masks.
    target['aff_mask'] = np.array(target['aff_mask'], dtype=np.uint8).reshape(height, width)
    target['aff_binary_masks'] = np.array(target['aff_binary_masks'], dtype=np.uint8).reshape(-1, height, width)

    # ids and bboxs.
    target['obj_ids'] = np.array(target['obj_ids'], dtype=np.int32).flatten()
    target['obj_boxes'] = np.array(target['obj_boxes'], dtype=np.int32).reshape(-1, 4)
    target['aff_ids'] = np.array(target['aff_ids'], dtype=np.int32).flatten()

    # depth images.
    # target['depth_8bit'] = np.squeeze(np.array(target['depth_8bit'], dtype=np.uint8))
    # target['depth_16bit'] = np.squeeze(np.array(target['depth_16bit'], dtype=np.uint16))

    return target

def draw_bbox_on_img(image, obj_ids, boxes, color=(255, 255, 255), scores=None):
    bbox_img = image.copy()

    if scores is None:
        for obj_id, bbox in zip(obj_ids, boxes):
            bbox = dataset_utils.format_bbox(bbox)
            # see dataset_utils.get_bbox for output of bbox.
            # x1,y1 ------
            # |          |
            # |          |
            # |          |
            # --------x2,y2
            bbox_img = cv2.rectangle(bbox_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, 1)

            cv2.putText(bbox_img,
                        f'{map_obj_id_to_name(obj_id)}',
                        (bbox[0], bbox[1] - 5),
                        cv2.FONT_ITALIC,
                        0.6,
                        color)
    else:
        for score, obj_id, bbox in zip(scores, obj_ids, boxes):
            bbox = format_bbox(bbox)
            # see dataset_utils.get_bbox for output of bbox.
            # x1,y1 ------
            # |          |
            # |          |
            # |          |
            # --------x2,y2
            bbox_img = cv2.rectangle(bbox_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, 1)

            cv2.putText(bbox_img,
                        f'{map_obj_id_to_name(obj_id)}: {score:.3f}',
                        (bbox[0], bbox[1] - 5),
                        cv2.FONT_ITALIC,
                        0.6,
                        color)

    return bbox_img

def get_segmentation_masks(image, obj_ids, binary_masks):

    height, width = image.shape[0], image.shape[1]
    instance_masks = np.zeros((height, width), dtype=np.uint8)
    instance_mask_one = np.ones((height, width), dtype=np.uint8)

    if len(binary_masks.shape) == 2:
        binary_masks = binary_masks[np.newaxis, :, :]

    for idx, obj_id in enumerate(obj_ids):
        binary_mask = binary_masks[idx, :, :]

        instance_mask = instance_mask_one * obj_id
        instance_masks = np.where(binary_mask, instance_mask, instance_masks).astype(np.uint8)

    return instance_masks

def colorize_aff_mask(instance_mask):

    instance_to_color = color_map_aff_id()
    color_mask = np.zeros((instance_mask.shape[0], instance_mask.shape[1], 3), dtype=np.uint8)
    for key in instance_to_color.keys():
        color_mask[instance_mask == key] = instance_to_color[key]

    return np.squeeze(color_mask)

def color_map_aff_id():
    ''' [red, blue, green]'''

    color_map_dic = {
    0:  [0, 0, 0],
    1:  [133, 17, 235],     # grasp: purple
    2:  [17, 235, 139],     # cut: teal
    3:  [235, 195, 17],     # scoop: yellow/gold
    4:  [17, 103, 235],     # contain: dark blue
    5:  [176, 235, 17],     # pound: light green/yellow
    6:  [76, 235, 17],      # support: green
    7:  [17, 235, 225],     # wrap-grasp: light blue
    }
    return color_map_dic
def format_bbox(bbox):
    return np.array(bbox, dtype=np.int32).flatten()