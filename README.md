# GCD


****<br>

Jinyan Xu:A Generalised New Method for Anomalous Phased Array Radar Echo Image Restoration Based on Generative Adversarial Network

## Introduction

__Generator.__ Image inpainting is cast into two subtasks, _i.e._, _structure-constrained texture synthesis_ (left, blue) and _texture-guided structure reconstruction_ (right, red), and the two parallel-coupled streams borrow encoded deep features from each other. The Bi-GFF module and CFA module are stacked at the end of the generator to further refine the results. 

__Discriminator.__ The texture branch estimates the generated texture, while the structure branch guides structure reconstruction.

<img src='assets/framework.png'/>

## Prerequisites

- Python >= 3.6
- PyTorch >= 1.0
- NVIDIA GPU + CUDA cuDNN

## Getting Started

### Installation

- Clone this repo:

```
git clone https://github.com/Xuyanyan88/GCDmodel

```
### Datasets

**Image Dataset.** 
**Mask Dataset.** 

### Training


```
python train.py \
  --image_root [path to image directory] \
  --mask_root [path mask directory]

python train.py \
  --image_root [path to image directory] \
  --mask_root [path to mask directory] \
  --pre_trained [path to checkpoints] \
  --finetune True
```

__Distributed training support.__ You can train model in distributed settings.

```
python -m torch.distributed.launch --nproc_per_node=N_GPU train.py
```

### Testing

To test the model, you run the following code.

```
python test.py \
  --pre_trained [path to checkpoints] \
  --image_root [path to image directory] \
  --mask_root [path to mask directory] \
  --result_root [path to output directory] \
  --number_eval [number of images to test]
```

## Citation


If any part of our paper and repository is helpful to your work, please generously cite with:

