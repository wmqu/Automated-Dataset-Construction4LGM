import os

import clip
import cv2
import torch
from PIL import Image
import numpy as np

def umd_predict_again(image_name, cut_images, predict_boxes, predict_classes, predict_scores):
    """
              Function:
                  Re-correction generated object classes of Faster RCNN
              Args:
                  1.image_name
                  2.cut_images: crop image file
                  3.predict_boxes
                  4.predict_classes
                  5.predict_scores
       """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_box = 0
    predict_boxes_list = []
    predict_classes_list = []
    predict_scores_list = []

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    labels = ["bowl", "cup", "hammer", "knife", "ladle", "mallet", "mug", "pot", "saw", "scissors", "scoop", "shears", "shovel", "spoon", "tenderizer", "trowel", "turner"]
    probs_box_thresh = 0.7
    for image in cut_images:
        img = preprocess(Image.open(image)).unsqueeze(0).to(device)
        text = clip.tokenize([f"This is a photo of a {label}" for label in labels]).to(device)
        with torch.no_grad():
            logits_per_image, logits_per_text = clip_model(img, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        probs_max_score = np.max(probs)
        probs_max_index = np.where(probs[0] == probs_max_score)[0][0]
        if predict_scores[num_box] > 0.9 or (probs_max_index == predict_classes[num_box] and predict_scores[
            num_box] + probs_max_score >= probs_box_thresh):
            predict_boxes_list.append(predict_boxes[num_box])
            predict_classes_list.append(predict_classes[num_box])
            predict_scores_list.append(predict_scores[num_box])
        elif probs_max_index != predict_classes[num_box] and probs_max_score > probs_box_thresh:
            predict_boxes_list.append(predict_boxes[num_box])
            predict_classes_list.append(probs_max_index)
            predict_scores_list.append(probs_max_score)
        num_box += 1
    return predict_boxes_list, predict_classes_list, predict_scores_list
