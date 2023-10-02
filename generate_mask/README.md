# generate_mask

### 1. Requirements

```bash
pip install -r requirements.txt
```
### 2. Available Architectures
Specify the model architecture with '--model ARCH_NAME' and set the output stride using '--output_stride OUTPUT_STRIDE'.

| DeepLabV3    |  DeepLabV3+        |
| :---: | :---:     |
|deeplabv3_resnet50|deeplabv3plus_resnet50|
|deeplabv3_resnet101|deeplabv3plus_resnet101|
|deeplabv3_mobilenet|deeplabv3plus_mobilenet |

All pretrained models: [Dropbox](https://www.dropbox.com/sh/w3z9z8lqpi8b2w7/AAB0vkl4F5vy6HdIhmRCTKHSa?dl=0), [Tencent Weiyun](https://share.weiyun.com/qqx78Pv5)

### 3. Load the pretrained model:
```python
model.load_state_dict( torch.load( CKPT_PATH )['model_state']  )
```
### 4. Visualize segmentation outputs:
```python
outputs = model(images)
preds = outputs.max(1)[1].detach().cpu().numpy()
colorized_preds = val_dst.decode_target(preds).astype('uint8') # To RGB images, (N, H, W, 3), ranged 0~255, numpy array
# Do whatever you like here with the colorized segmentation maps
colorized_preds = Image.fromarray(colorized_preds[0]) # to PIL Image
```

### 5. Atrous Separable Convolution

**Note**: pre-trained models in this repo **do not** use Seperable Conv.

Atrous Separable Convolution is supported in this repo. We provide a simple tool ``network.convert_to_separable_conv`` to convert ``nn.Conv2d`` to ``AtrousSeparableConvolution``. **Please run main.py with '--separable_conv' if it is required**. See 'main.py' and 'network/_deeplab.py' for more details. 


### 6. Performance on IIT-AFF (9 affordance) and UMD (7 affordance)

Training: 256x256 random crop  
validation: 256x256 center crop

Model: DeepLabV3-ResNet50

|  dataset          |  Overall Acc  | Mean Acc  | FreqW Acc| mIoU      |
| :--------        | :--------: | :--------:    | :----:     | :----: |
|    IIT-AFF      |    0.922     |    0.738    | 0.859 |   0.656     | 
|    UMD      |   0.994    |    0.940    | 0.988 |   0.878     |  


### 7. Training and Testing

#### 7.1 Training
Run main.py for training and validation.

#### 7.2 Continue training

Run main.py with '--continue_training' to restore the state_dict of optimizer and scheduler from YOUR_CKPT.

```bash
python main.py ... --ckpt YOUR_CKPT --continue_training
```

#### 7.3 generate single object mask

Run predict.py and use the generated weights to predict the cropped regions’ part-level affordance mask. 




## Reference

[1] [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

[2] [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)
