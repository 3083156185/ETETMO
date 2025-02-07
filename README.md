<div align="center">
<h1> 🦖ETETMO </h1>
<h3>ETETMO: An End-to-End Visual Intelligent Surveillance Framework for Robust Tracking of Moving Objects on the Airport Surface</h3>
[XIANGQING DONG](https://orcid.org/0009-0004-3789-4359)<sup>1</sup>,[MENG DING](https://faculty.nuaa.edu.cn/dm/zh_CN/index.htm)<sup>:email:</sup><sup>1</sup>,[YUBIN XU]<sup>2</sup>, [YIMING XU]<sup>1</sup>, [LI CAO]<sup>1</sup> 

<sup>1</sup> Nanjing University of Aeronautics and Astronautics, <sup>2</sup>Chinese Academy of Civil Aviation Science and Technology
  
<sup>:email:</sup> corresponding author.
</div>

## :fire: Updates

- **`2/07/2025`**: We release the pre-trained models and the detect code.

- **`1/07/2025`**: We have published some visualization experiments static and dynamic results.

- **`11/02/2024`**: We submitted the paper. Code and pre-trained model are coming soon.

## :rocket: Introduction
This project contains the official PyTorch implementation, pre-trained models, fine-tuning code, and detect demo for ETETMO.

* An end-to-end intelligent visual surveillance framework,ETETMO, is proposed for robust tracking of moving objects on the airport surface.

* ETETMO replaces the traditional manually designed ID association methods with a learnable ID classification strategy.

* ETETMO surpasses state-of-the-art methods across five key evaluation metrics, achieving a 7.6% improvement in HOTA and an 11% improvement in IDF1.

## :page_facing_up: Overview

<img src="result/image2.jpg" width="800">

## :sparkles: Model Zoo
* [Weight CKPT](https://huggingface.co/hao9610/OV-DINO/resolve/main/ovdino_swint_ogc-coco50.2_lvismv40.1_lvis32.9.pth) 
## :checkered_flag: Getting Started
### 1. DATASET Structure
```
ETETMO
├── dataset
│   ├── ASV-T2024
│   │   ├── annotations
│   │   ├── train
│   │   ├── val
│   │   └── test
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   └── val2017
```
### 2. Installation
...
## :computer: Visualization of experimental results

<img src="result/1.gif" width="800">

<img src="result/image7.png" width="800">

## :blush: Acknowledge
This project has referenced some excellent open-sourced repos ([Detectron2](https://github.com/facebookresearch/detectron2), [detrex](https://github.com/IDEA-Research/detrex), [RT-DETR](https://github.com/lyuwenyu/RT-DETR)). Thanks for their wonderful works and contributions to the community.

