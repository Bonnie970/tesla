CUDA_VISIBLE_DEVICES=0 python matting_unpool.py \
	--alpha_path=/home/vortex/bonniehu/data/alpha2/\
	--trimap_path=/home/vortex/bonniehu/data/trimap2/\
	--fg_path=/home/vortex/bonniehu/data/fg2/\
	--bg_path=/home/vortex/bonniehu/data/bg2/\
	--rgb_path=/home/vortex/bonniehu/data/rgb2/\
	--model_path=/home/vortex/bonniehu/Segmentation-Refinement/vgg16_weights.npz\
	--log_dir=/home/vortex/bonniehu/Segmentation-Refinement/log\
	--save_ckpt_path=/home/vortex/bonniehu/Segmentation-Refinement/ckpt\
	--save_meta_path=/home/vortex/bonniehu/Segmentation-Refinement/meta/my-model.meta\
	--dataset_name=Adobe \
	--image_height=320 \
	--image_width=320 \
	--max_epochs=50 \
	--batch_size=8 \
	--learning_rate=0.0001 \
	--learning_rate_decay=0.25 \
	--learning_rate_decay_steps=5000 \
	--restore_from_ckpt=False \
	--save_log_steps=150 \
	--save_ckpt_steps=150
