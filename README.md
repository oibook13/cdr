# Centroid-based Dimensionality Reduction
Code for Centroid-based Dimensionality Reduction module

## Test Environment
* Ubuntu 16.04
* Python 3.6
* PyTorch 0.4.1

## Requirement
Install pytorch, e.g., with conda
```bash
conda install pytorch torchvision -c pytorch
```
Download [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/)

## Usage
To use CDR-2-R to classify images, run
```bash
export checkpoint_dir='path_to_checkpoint_parent_folder/checkpoints/tinyimagenet/resnet101_cdr_pbl'
CUDA_VISIBLE_DEVICES=0,1 python train_timgnet_models.py \
	--model resnet101 \
	--pretrained \
	--train_batch 64 \
	--epochs 30 \
	--lr 1e-03 \
	--lr_decay 0.1 \
	--lr_scheduler 10 20 \
	--dataset 'path_to_tinyimagenet_parent_folder/tinyimagenet/images/' \
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
Specifically, you need to set checkpoint folder env variable **checkpoint_dir** and dataset folder **dataset**. To use CDR-2-M for classification, set **cdr_num_init_samples** to 100.

To use fully-connected layer as a baseline, run
```bash
export checkpoint_dir='path_to_checkpoint_parent_folder/checkpoints/tinyimagenet/resnet101_fc'
CUDA_VISIBLE_DEVICES=0,1 python train_timgnet_models.py \
    --model resnet101 \
    --pretrained \
    --train_batch 64 \
    --epochs 30 \
    --lr 1e-03 \
    --lr_decay 0.1 \
    --lr_scheduler 10 20 \
    --dataset 'path_to_tinyimagenet_parent_folder/tinyimagenet/images/' \
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
