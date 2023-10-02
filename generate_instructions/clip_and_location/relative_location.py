# coding=gbk
import numpy as np


def center_of_bbox(bbox):
    return [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]


def area_of_bbox(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def relative_spatial_location(predict_boxes, image_size):
    """
        Function:
            1.Left/right/middle
            2.Top/bottom
            3.Front/back
        Args:
            1.predict_boxes
            2.image_size: (h, w)
    """
    # left/right/middle
    # top/bottom
    horizontal_thresh = 50
    vertical_thresh = 50
    if len(predict_boxes) > 1:
        tmp_bbox_center = []
        for ind in range(len(predict_boxes)):
            tmp_bbox_center.append(center_of_bbox(predict_boxes[ind]))
        tmp_bbox_center = np.array(tmp_bbox_center)
        xmin_ind, xmax_ind = np.argmin(tmp_bbox_center[:, 0]), np.argmax(tmp_bbox_center[:, 0])
        ymin_ind, ymax_ind = np.argmin(tmp_bbox_center[:, 1]), np.argmax(tmp_bbox_center[:, 1])
        if (tmp_bbox_center[xmax_ind, 0] - tmp_bbox_center[
            xmin_ind, 0]) > horizontal_thresh:  # avoid the little shift of bbox
            predict_boxes[xmin_ind].append('left')
            predict_boxes[xmax_ind].append('right')
            if len(predict_boxes) == 3:   # 三个目标
                for ind in range(len(predict_boxes)):
                    if ind not in [xmin_ind, xmax_ind]:
                        predict_boxes[ind].append('middle')
            elif len(predict_boxes) > 3:  # 三个以上的目标
                for ind in range(len(predict_boxes)):
                    if ind not in [xmin_ind, xmax_ind]:
                        if tmp_bbox_center[ind, 0] > image_size[1] * 3 / 4:  # image_size：h*w
                            predict_boxes[ind].append('right')
                        elif tmp_bbox_center[ind, 0] < image_size[1] / 4:
                            predict_boxes[ind].append('left')
                        else:
                            predict_boxes[ind].append('middle')
            else:
                pass

        if (tmp_bbox_center[ymax_ind, 1] - tmp_bbox_center[
            ymin_ind, 1]) > vertical_thresh:  # avoid the little shift of bbox
            predict_boxes[ymax_ind].append('bottom')
            predict_boxes[ymin_ind].append('top')
            for ind in range(len(predict_boxes)):
                if ind not in [ymin_ind, ymax_ind]:
                    if tmp_bbox_center[ind, 1] > image_size[0] * 3 / 4:
                        predict_boxes[ind].append('bottom')
                    elif tmp_bbox_center[ind, 1] < image_size[1] / 4:
                        predict_boxes[ind].append('top')
                    else:
                        pass

        ### front/behind
        # area_ratio_thresh_low = 0.4
        # area_ratio_thresh_up = 0.8
        # for key, value in descriptor.items():
        #     if len(value) > 1:
        #         tmp_bbox_area = []
        #         for ind in range(len(value)):
        #             tmp_bbox_area.append(area_of_bbox(value[ind]['bbox']))
        #         tmp_bbox_area = np.array(tmp_bbox_area)
        #         min_ind, max_ind = np.argmin(tmp_bbox_area), np.argmax(tmp_bbox_area)
        #         if tmp_bbox_area[min_ind] / tmp_bbox_area[max_ind] < area_ratio_thresh_low:
        #             descriptor[key][min_ind]['spatial'].append('behind')
        #             descriptor[key][max_ind]['spatial'].append('front')
        #             if len(value) > 3:
        #                 for ind in range(len(value)):
        #                     if ind not in [min_ind, max_ind]:
        #                         if tmp_bbox_area[ind] / tmp_bbox_area[max_ind] < area_ratio_thresh_low:
        #                             descriptor[key][ind]['spatial'].append('behind')
        #                         elif tmp_bbox_area[ind] / tmp_bbox_area[max_ind] > area_ratio_thresh_up:
        #                             descriptor[key][ind]['spatial'].append('front')
        #                         else:
        #                             pass

    return predict_boxes

# if __name__ == '__main__':
#     predict_boxes = [[430, 512, 618, 574], [529, 413, 582, 537], [341, 430, 505, 554]]
#     boxes = relative_spatial_location(predict_boxes,[1024,1024])
#     print(boxes)
