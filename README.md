# Centroid-based Dimensionality Reduction
Code for Centroid-based Dimensionality Reduction module

## Test Environment
* Ubuntu 16.04
* Python 3.6
* PyTorch 0.4.1

## Requirement
1. Install pytorch, e.g., with conda
```bash
conda install pytorch torchvision -c pytorch
```
2. Download [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/).
Our directory structure is as follows
```
tinyimagenet/
├── test
├── train
└── val
```

## Usage
To use CDR-2-R to classify images, run
```bash
export checkpoint_dir='checkpoints/tinyimagenet/resnet101_cdr'
CUDA_VISIBLE_DEVICES=0,1 python train_timgnet_models.py \
	--model resnet101 \
	--pretrained \
	--train_batch 64 \
	--epochs 30 \
	--lr 1e-03 \
	--lr_decay 0.1 \
	--lr_scheduler 10 20 \
	--dataset '/home/xxx/project/dataset/tinyimagenet/' \
	--num_class 200 \
	--input_img_size 224 \
	--drm_type 'cdr' \
	--cdr_alpha 0.005 \
	--cdr_normalized_radius 1 \
	--cdr_t 0.05,0.05 \
	--cdr_p 1.5,2 \
	--cdr_num_init_samples -1 \
	--checkpoint $checkpoint_dir
```
Specifically, you need to set env variable **dataset** to your local dataset folder. To use CDR-2-M for classification, set **cdr_num_init_samples** to 100 and -1 indicates initialization using random numbers.

To use fully-connected layer as a baseline, run
```bash
export checkpoint_dir='checkpoints/tinyimagenet/resnet101_fc'
CUDA_VISIBLE_DEVICES=0,1 python train_timgnet_models.py \
    --model resnet101 \
    --pretrained \
    --train_batch 64 \
    --epochs 30 \
    --lr 1e-03 \
    --lr_decay 0.1 \
    --lr_scheduler 10 20 \
    --dataset '/home/xxx/project/dataset/tinyimagenet/' \
    --num_class 200 \
    --input_img_size 224 \
    --drm_type 'fc' \
    --fc_kaiming_alpha 0.0 \
    --checkpoint $checkpoint_dir
```

## Cite
```
@article{anonymouspaper,
  title={Centroid-based Dimensionality Reduction Module for Image Classification},
  author={XXX},
  year={2018}
}
```
