# Presentation of Datasets

## 1. IIT-AFF Dataset
### 1.1 IIT-AFF Dataset File Directory

```
    /IIT_Affordance 
        /train 
            /mask_object #  <=store affordances labels at pixel-level for each image
            /rgb # <= store original rgb image
            /single_object_mask # <= store cropped rgb image by Faster-RCNN and CLIP
                /00_00000090 # <= The folder name is the image name 
                    /00_00000090_0_0.png # <= single object affordance mask, the first 0 corresponds to the index of the value array in the JSON file, the second 0 represents the index of the object’s explicit instruction array.
                    /00_00000090_1_1.png
                    /00_00000090_2_0.png
                    /00_00000090_2_1.png
                ...    
            /xml # <= store locations of objects in each image
            /iit_train.json # <= store the instructions and box corresponding to the objects in the image
        ...
```
### 1.2. iit_train.json
take image 00_00000090.jpg as an example
```json
 {"00_00000090.jpg": [
    {
      "box": [
        345,
        346,
        585,
        523
      ],
      "explicit_sentence": [
        "Bring me the racket on the left and bottom to play",
        "grasp the racket on the left and bottom"
      ],
      "guid": 0,
      "implicit_sentence": "I want something to play",
      "label": "racket"
    },
    {
      "box": [
        467,
        284,
        519,
        391
      ],
      "explicit_sentence": [
        "Pass me the knife on the middle and top to cut",
        "grasp the knife on the middle and top"
      ],
      "guid": 1,
      "implicit_sentence": "I need something to cut",
      "label": "knife"
    },
    {
      "box": [
        553,
        267,
        627,
        415
      ],
      "explicit_sentence": [
        "Pass me the hammer on the right to beat",
        "grasp the hammer on the right"
      ],
      "guid": 2,
      "implicit_sentence": "An object that can beat",
      "label": "hammer"
    }
  ]}
```
## 2. UMD Dataset
### 2.1 UMD Dataset File Directory

```
    /UMD
        /train 
            /mask_object #  <=store affordances labels at pixel-level for each image
            /rgb # <= store original rgb image
            /single_object_mask # <= store cropped rgb image by Faster-RCNN and CLIP
                /bowl_01_00000002_rgb # <= The folder name is the image name 
                    /bowl_01_00000002_rgb_0_0.png # <= single object affordance mask, the first 0 corresponds to the index of the value array in the JSON file, the second 0 represents the index of the object’s explicit instruction array.
                ... 
            /umd_train.json # <= store the instructions and box corresponding to the objects in the image
        ...
```
### 2.2. umd_train.json
take image 00_00000090.jpg as an example
```json
{"bowl_01_00000002_rgb.jpg": [
    {
      "box": [
        247, 
        181,
        368, 
        261
       ],
      "explicit_sentence": [
        "Pass me the bowl to contain", 
        ""
       ],
      "guid": 0, 
      "implicit_sentence": "Give me an item that can contain",
      "label": "bowl"
    }]
 }
```