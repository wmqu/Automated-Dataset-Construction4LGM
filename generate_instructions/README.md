# generate instructions and bounding box

## The source code of FastRCNN in the pytorch official torchvision module
* https://github.com/pytorch/vision/tree/master/torchvision/models/detection


## Environment configuration：
* Python3.6/3.7/3.8
* Pytorch1.7.1(Note：must be 1.6.0 or above)
* pycocotools(Linux:`pip install pycocotools`; Windows:`pip install pycocotools-windows`)
* For detailed environment configuration, see`requirements.txt`

## File structure：
```
  ├── backbone: Feature Extraction Network
  ├── network_files: Faster R-CNN Network（include Fast R-CNN and RPN）
  ├── train_utils: Training and verification related modules
  ├── clip_and_location: Clip the object in the image and get the relative positional relationship of the object in the image
  ├── dataloader: Custom dataloader is used to read IIT-AFF and UMD dataset
  ├── openprompt: OpenPrompt toolkit
  ├── train_mobilenet.py: Train with MobileNetV2 as the backbone
  ├── train_resnet50_fpn.py: Train with resnet50+FPN as the backbone 
  ├── generate_instructions.py: Simple prediction script, use the trained weights for generating multi-type, multi-hierarchies instructions about different dataset based on prompt learning
  ├── iit_prompt.py: Generating affordance about IIT-AFF dataset based on prompt learning
  ├── umd_prompt.py: Generating affordance about UMD dataset based on prompt learning
  ├── iit_object.json: IIT-AFF dataset classes label file
  └── umd_object.json: UMD dataset classes label file
```
## Prepare Datasets
* IIT-AFF dataset is made up of 8,835 real-world images with corresponding locations of objects and affordances labels
at pixel-level for each image. This dataset has 10 object categories and 9 affordance classes. [IIT-AFF dataset download link](https://sites.google.com/site/iitaffdataset/)
* UMD dataset consists of 28843 RGB-D images for 105 kitchen, workshop, and gardening tools. This dataset does not
provide the location of the object bounding boxes, we calculate the coordinates of the object bounding boxes by using the
affordance mask. The dataset contains 17 object categories and 7 affordance classes. [UMD dataset download link](https://users.umiacs.umd.edu/~fer/affordance/Affordance.html)

For each dataset, the dataset is randomly divided into training, validation and testing sets with ratio 8:1:1.

## Pre-training weight download address（After downloading, put it in the backbone folder）：
In this experiment, ResNet50 is used.
* MobileNetV2 weights(renamed to`mobilenet_v2.pth`): https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
* Resnet50 weights(renamed to`resnet50.pth`): https://download.pytorch.org/models/resnet50-0676ba61.pth
* ResNet50+FPN weights(renamed to`fasterrcnn_resnet50_fpn_coco.pth`): https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth

## OpenPrompt
* The tool offers a lot of templates, we use the manual template
* The template is '{"placeholder":"text_a"}.In other words, give me something to {"mask"}'
* Define a pre-trained language model, we use the gpt2 as a pre-trained language model
* A `Verbalizer` is another important (but not necessary) in prompt-learning, which projects the original labels (we have defined them as `classes`) to a set of label words.

## Positional Relationship
The spatial relationship of two dimensions: horizontal (left, middle and right), vertical (top and bottom).

## Training
Load the pretrained model:
```python
if load_pretrain_weights:
    # https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
    weights_dict = torch.load("", map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
```
   if use resnet50 as backbone, you can run the following commands:
   ```
    python train_res50_fpn.py --data_path XXX --dataset {dataset_name} --num_classes 10 --batch_size 2 
   ```    
   if use mobilenet_v2 as backbone, you can run the following commands:

   ```
    python train_mobilenetv2.py --data_path XXX --dataset {dataset_name} --num_classes 10 --batch_size 2 
   ```
    
   We set --num_classes(No background included) 10 for IIT-AFF dataset, and --num_classes 17 for UMD dataset. 
## generate instructions
   ```
    python generate_instructions.py --image_file_path XXX --json_file_path XXX --dataset {dataset_name} --num_classes 11 --crop_path XXX --weights_path XXX --label_json_path XXX --no_object_txt XXX --error_detect_txt XXX --error_affordance_txt XXX
   ```

   We set --num_classes(background included) 11 for IIT-AFF dataset, and --num_classes 18 for UMD dataset. 
##  Apply to other datasets
#### Step 1: train othor dataset.
  If you want to generate dataset with bounding boxes and explicit and implicit affordance instructions through the model, you must first train dataset though FastRCNN.
#### Step 2: Modify iit_prompt.py.
  * Modify the classes according to the affordance in your dataset
```python
classes = [ # There are seven affordance classes in IIT-AFF dataset
    "contain",
    "cut",
    "display",
    "engine",
    "hit",
    "support",
    "pound"
]
```

   * Modify the Verbalizer according to the affordance in your dataset, which projects the original labels (we have defined them as `classes`, remember?) to a set of label words.
```python
from openprompt.prompts import ManualVerbalizer
label_words = {
    "contain": ["contain", "drink", "pour", "cook"],
    "cut": ["cut"],
    "display": ["show", "observe", "watch"],
    "engine": ["engine", "operate"],
    "pound": ["hit", "strike", "beat"],
    "support": ["smooth", "stir fry", "support"],
    "hit": ["swing", "play"]
}

promptVerbalizer = ManualVerbalizer(
    classes=classes,
    label_words=label_words,
    tokenizer=tokenizer,
)
```
#### Step 3: Modify iit_clip_image.py.
  Modify labels in the CLIP model
```python
labels = ["bowl", "tvm", "pan", "hammer", "knife", "cup", "drill", "racket", "spatula", "bottle"]
```
#### Step 4: run generate_instructions.py.
## Reference

[1] [Object-Based Affordances Detection with Convolutional Neural Networks and Dense Conditional Random Fields](https://www.csc.liv.ac.uk/~anguyen/assets/pdfs/2017_IROS_CRFSeg.pdf)

[2] [Affordance Detection of Tool Parts from Geometric Features](https://www.researchgate.net/profile/Cornelia-Fermueller/publication/282930046_Affordance_detection_of_tool_parts_from_geometric_features/links/565dd4c608aefe619b26bad8/Affordance-detection-of-tool-parts-from-geometric-features.pdf)

[3] [OpenPrompt: An Open-source Framework for Prompt-learning](https://arxiv.org/pdf/2111.01998.pdf)

