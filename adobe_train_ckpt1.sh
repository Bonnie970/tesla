CUDA_VISIBLE_DEVICES=0 python matting_unpool.py \
	--alpha_path=/home/vortex/bonniehu/Adobe_data/processed_data_for_train/alpha_1 \
	--trimap_path=/home/vortex/bonniehu/Adobe_data/processed_data_for_train/trimap_1 \
	--fg_path=/home/vortex/bonniehu/Adobe_data/processed_data_for_train/fg_1 \
	--bg_path=/home/vortex/bonniehu/Adobe_data/processed_data_for_train/bg_1 \
	--rgb_path=/home/vortex/bonniehu/Adobe_data/processed_data_for_train/rgb_1 \
	--model_path=/home/vortex/bonniehu/Segmentation-Refinement/vgg16_weights.npz\
	--log_dir=/home/vortex/bonniehu/Segmentation-Refinement/log\
	--save_ckpt_path=/home/vortex/bonniehu/Segmentation-Refinement/ckpt_1\
	--save_meta_path=/home/vortex/bonniehu/Segmentation-Refinement/meta/my-model.meta\
	--dataset_name=Adobe \
	--image_height=320 \
	--image_width=320 \
	--max_epochs=1 \
	--batch_size=4 \
	--learning_rate=0.0001 \
	--learning_rate_decay=0.25 \
	--learning_rate_decay_steps=25000 \
	--restore_from_ckpt=False \
	--save_log_steps=50 \
	--save_ckpt_steps=1500
