CUDA_VISIBLE_DEVICES=0 python matting_unpool.py \
	--alpha_path=/home/vortex/bonniehu/Adobe_data/Training_set/composed_alpha \
	--trimap_path= \
	--fg_path=/home/vortex/bonniehu/Adobe_data/Training_set/composed_fg \
	--bg_path=/home/vortex/bonniehu/Adobe_data/Training_set/composed_bg \
	--rgb_path=/home/vortex/bonniehu/Adobe_data/Training_set/composed_images \
	--model_path=/home/vortex/bonniehu/Segmentation-Refinement/vgg16_weights.npz\
	--log_dir=/home/vortex/bonniehu/Segmentation-Refinement/log_wl=0.25\
	--save_ckpt_path=/home/vortex/bonniehu/Segmentation-Refinement/ckpt_wl=0.25\
	--save_meta_path=/home/vortex/bonniehu/Segmentation-Refinement/meta/my-model-wl=0.25.meta\
	--dataset_name=Adobe \
	--image_height=320 \
	--image_width=320 \
	--max_epochs=1000 \
	--batch_size=4 \
	--learning_rate=0.0001 \
	--learning_rate_decay=0.5 \
	--learning_rate_decay_steps=25000 \
	--restore_from_ckpt=True \
	--save_log_steps=50 \
	--save_ckpt_steps=2000
