import torch
import torchvision
import torch.utils.data
from pycocotools.coco import COCO


def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0
    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        hw, targets = ds.coco_index(img_idx)
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict['id'] = image_id
        img_dict['height'] = hw[0]
        img_dict['width'] = hw[1]
        dataset['images'].append(img_dict)
        bboxes = targets["boxes"]
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets['labels'].tolist()
        areas = targets['area'].tolist()
        # iscrowd = targets['iscrowd'].tolist()
        iscrowd = targets['iscrowd']
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann['image_id'] = image_id
            ann['bbox'] = bboxes[i]
            ann['category_id'] = labels[i]
            categories.add(labels[i])
            ann['area'] = areas[i]
            ann['iscrowd'] = iscrowd[i]
            ann['id'] = ann_id
            dataset['annotations'].append(ann)
            ann_id += 1
    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds

# dataloader = r"E:\jinxiao\dataloader\IIT_Affordances_2017\val.txt"
def get_coco_api_from_dataset(dataset):
    for _ in range(17):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)

#
# def main():
#     nw = min([os.cpu_count(), 2 if 2 > 1 else 0, 8])  # number of workers
#     print('Using %g dataloader workers' % nw)
#     data_transform = {
#         "train": transforms.Compose([transforms.ToTensor(),
#                                      transforms.RandomHorizontalFlip(0.5)]),
#         "val": transforms.Compose([transforms.ToTensor()])
#     }
#     val_dataset = VOCDataSet(r"E:\jinxiao\dataloader\IIT_Affordances_2017", data_transform["val"], "val.txt")
#     val_data_set_loader = torch.utils.data.DataLoader(val_dataset,
#                                                       batch_size=1,
#                                                       shuffle=False,
#                                                       pin_memory=True,
#                                                       num_workers=nw,
#                                                       collate_fn=val_dataset.collate_fn)
#     get_coco_api_from_dataset(val_data_set_loader.dataloader)
#
# if __name__ == "__main__":
#     main()
