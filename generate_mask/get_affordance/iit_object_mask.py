
import json
import shutil

import numpy as np

from ..datasets import IIT

from generate_mask.utils import *


def ergodic_matrix(slice_matrix, pred, aff_id):
    row = pred.shape[0]
    col = pred.shape[1]
    for i in range(0, row):
        for j in range(0, col):
            if pred[i][j] == aff_id:
                slice_matrix[i][j] = aff_id
    return slice_matrix
def get_object_mask(img_name, pred, base_path, save_dir_name, save_val_results_to):
    split_image_name = img_name.strip().split("_")
    num_object = int(split_image_name[-1:][0])
    img_suffix = save_dir_name + ".jpg"
    init_image_path = base_path + '/' + img_suffix
    decode_fn = IIT.decode_target
    save_del_index = []
    # json file path
    with open(r'', 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
        init_matrix = getImageMatrix(init_image_path)
        image_info = json_data[img_suffix]
        print(img_suffix, num_object)

        xmin = image_info[num_object]["box"][0]
        ymin = image_info[num_object]["box"][1]
        xmax = image_info[num_object]["box"][2]
        ymax = image_info[num_object]["box"][3]
        slice_matrix = init_matrix[ymin:ymax, xmin:xmax]
        obj_id = iit_map_obj_name_to_id(json_data[img_suffix][num_object]["label"])
        aff_ids = iit_map_obj_id_to_aff_id(obj_id)
        assign_matrix = []
        pre_aff_ids = np.unique(pred)[1:]
        if len(pre_aff_ids) == 3:
            for aff_id in aff_ids[0]:
                if aff_id != 5 and aff_id in pre_aff_ids.tolist():  # If the number of predicted affordance is greater than 3, then ignore grasp affordance
                    assign_matrix.append(ergodic_matrix(slice_matrix, pred, aff_id))
                    slice_matrix = np.zeros((pred.shape[0], pred.shape[1]), dtype=np.uint8)
            # image_info[num_object]["explicit_sentence"][1] = ""  # Delete the corresponding grasp sentence
        else:
            for aff_id in aff_ids[0]:
                assign_matrix.append(ergodic_matrix(slice_matrix, pred, aff_id))
                slice_matrix = np.zeros((pred.shape[0], pred.shape[1]), dtype=np.uint8)
        for i in range(0, len(assign_matrix)):
            init_matrix = getImageMatrix(init_image_path)
            if np.all(assign_matrix[i] == 0):
                save_del_index.append(i)
                continue
            init_matrix[ymin:ymax, xmin:xmax] = assign_matrix[i]
            single_image = Image.fromarray(init_matrix.astype('uint8'))
            # colorized_preds = decode_fn(init_matrix).astype('uint8')
            # colorized_preds = Image.fromarray(colorized_preds)
            if save_val_results_to:
                save_dir_path = os.path.join(save_val_results_to, save_dir_name)
                mkdir(save_dir_path)
                single_image.save(os.path.join(save_dir_path, '{}_{}.png'.format(img_name, i)))
        # counter = 0
        # for index in save_del_index:
        #     index = index - counter
        #     image_info[num_object]["explicit_sentence"][index] = ""
        #     counter += 1
        # iit json path
        # with open(r'', 'w', encoding='utf8')as f:
        #     dic_to_json = json.dumps(eval(str(json_data)))
        #     json_data.update(eval(dic_to_json))
        #     json.dump(json_data, f)






