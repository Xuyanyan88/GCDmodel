# GCD


****<br>

Jinyan Xu:A Generalised New Method for Anomalous Phased Array Radar Echo Image Restoration Based on Generative Adversarial Network

## Introduction

Research on X-band phased array radar image restoration


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
Xu, . jinyan . (2025). PPI image of an X-band phased array radar [Data set]. Zenodo. https://doi.org/10.5281/zenodo.16732041
@InProceedings{Guo_2021_ICCV,
    author    = {Guo, Xiefan and Yang, Hongyu and Huang, Di},
    title     = {Image Inpainting via Conditional Texture and Structure Dual Generation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {14134-14143}
}
