export checkpoint_dir='checkpoints/tinyimagenet/resnet101_fc'
CUDA_VISIBLE_DEVICES=1,2 python train_timgnet_models.py \
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
export checkpoint_dir='checkpoints/tinyimagenet/resnet101_cdr_pbl'
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
export checkpoint_dir='checkpoints/tinyimagenet/resnet101_cdr_pbl'
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
	--cdr_num_init_samples 100 \
	--checkpoint $checkpoint_dir