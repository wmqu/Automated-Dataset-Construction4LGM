# coding=gbk
import argparse
import time
import shutil

import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms

from generate_instructions.clip_and_location.iit_clip import iit_predict_again
from generate_instructions.clip_and_location.umd_clip import umd_predict_again
from generate_instructions.network_files import FasterRCNN
from generate_instructions.backbone import resnet50_fpn_backbone
from generate_instructions.clip_and_location.relative_location import relative_spatial_location
from generate_instructions.openprompt.iit_prompt import predict_affordance

import json
import clip
import os
import numpy as np

from generate_instructions.openprompt.umd_prompt import predict_umd_affordance

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options

    parser.add_argument("--image_file_path", type=str, default=r"", help="path to a single image or image directory")
    parser.add_argument("--json_file_path", type=str,  default="", help="path to save json file")
    parser.add_argument('--num_classes', default=11, type=int, help='num_classes(background included)')
    parser.add_argument('--dataset', type=str, default='umd', choices=['iit', 'umd'], help='Name of dataset')
    parser.add_argument("--crop_path", type=str, default='', help='path to crop image directory')
    parser.add_argument("--weights_path", type=str, default='', help='path to weights')
    parser.add_argument("--label_json_path", type=str, default=r'./dataloader/iit_object.json', help='path to label json')
    parser.add_argument("--no_object_txt", type=str, default='',
                        help='No object was predicted, save the image name to a txt file')
    parser.add_argument("--error_detect_txt", type=str, default='',
                        help='Detected object category error, save the image name to a txt file')
    parser.add_argument("--error_affordance_txt", type=str, default='',
                        help='Predicted error affordance, save the image name to a txt file')
    return parser


def create_model(num_classes):
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def crop(cut_path, image_name, image_file_path, bboxs):
    img = cv2.imread(image_file_path)
    if not os.path.exists(cut_path):
        os.makedirs(cut_path)
    else:
        print("Empty folder")
        shutil.rmtree(cut_path)
        os.mkdir(cut_path)
    obj_i = 0

    for b in bboxs:
        img_cut = img[int(b[1]):int(b[3]), int(b[0]):int(b[2]), :]
        try:
            cv2.imwrite(os.path.join(cut_path, '{}_{}.jpg'.format(image_name, obj_i)), img_cut)
        except:
            continue
        obj_i += 1

    cut_img = []
    for i in os.listdir(cut_path):
        cut_img.append(cut_path + '/' + i)
    return cut_img


def main():
    # get devices
    opts = get_argparser().parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    model = create_model(opts.num_classes)

    # load train weights
    assert os.path.exists(opts.weights_path), "{} file dose not exist.".format(opts.weights_path)
    weights_dict = torch.load(opts.weights_path, map_location='cpu')
    # weights_dict = weights_dict[0]["model"] if "model" in weights_dict[0] else weights_dict
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict[0]
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict  iit
    assert os.path.exists(opts.label_json_path), "json file {} dose not exist.".format(opts.label_json_path)
    with open(opts.label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    # load image
    for file in os.listdir(opts.image_file_path):
        image_name = os.path.basename(opts.image_file_path)
        image_path = opts.image_file_path + '/' + file
        original_img = Image.open(image_path)
        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()
        with torch.no_grad():
            # init
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            if len(predict_boxes) == 0:
                print("No targets detected!")
                # write into txt
                with open(opts.no_object_txt, "a") as f:
                    i_name = file.split('.')[0]
                    f.write(i_name + "\n")
            else:
                image_name = file.split('.')[0]
                cut_path = opts.crop_path + image_name
                # Crop the predicted picture
                cut_images = crop(cut_path, image_name, image_path, predict_boxes)
                # CLIP model
                if opts.dataset == "iit":
                    predict_boxes_list, predict_classes_list, predict_scores_list = iit_predict_again(image_name,
                                                                                                  cut_images,
                                                                                                  predict_boxes,
                                                                                                  predict_classes,
                                                                                                  predict_scores)
                elif opts.dataset == "umd":
                    predict_boxes_list, predict_classes_list, predict_scores_list = umd_predict_again(image_name,
                                                                                                  cut_images,
                                                                                                  predict_boxes,
                                                                                                  predict_classes,
                                                                                                  predict_scores)
                if len(predict_boxes_list) != 0:
                    new_predict_boxes = np.array(predict_boxes_list)
                    new_predict_classes = np.array(predict_classes_list)
                    new_predict_scores = np.array(predict_scores_list)
                    cut_images_again = crop(cut_path, image_name, image_path, new_predict_boxes)
                    boxes_location = relative_spatial_location(new_predict_boxes.tolist(), img.shape[-2:])


                    classes = []
                    for object_ind in predict_classes_list:
                        classes.append(category_index[str(object_ind)])
                    if opts.dataset == "iit":
                        affordance_queries = predict_affordance(classes, boxes_location)
                    elif opts.dataset == "umd":
                        affordance_queries = predict_umd_affordance(classes, boxes_location)
                    if affordance_queries != []:
                        new_dic = {
                            file: affordance_queries
                        }

                        with open(opts.json_file_path, 'r', encoding='utf8')as fp:
                            json_data = json.load(fp)
                            with open(opts.json_file_path, 'w') as f:
                                dic_to_json = json.dumps(eval(str(new_dic)))
                                json_data.update(eval(dic_to_json))
                                json.dump(json_data, f)
                    else:
                        print(file + "Affordance detection is inaccurate!")
                        shutil.rmtree(cut_path)
                        # write into txt
                        with open(opts.error_affordance_txt, "a") as f:
                            txt = image_name
                            f.write(txt + "\n")

                else:
                    print(file + "The picture was not detected accurately!")
                    # write into txt
                    with open(opts.error_detect_txt, "a") as f:
                        txt = image_name
                        f.write(txt + "\n")
                    shutil.rmtree(cut_path)


if __name__ == '__main__':
    main()
