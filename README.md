######################
### data compositing 
#python3

cd ~/Adobe_data/Training/
source activate mypython3
cat data_stats 

#change dir names on top of Compositing_code.py 
#composite 'Other' and 'Adobe_licensed' separately 
#make sure COCO is downloaded somewhere -> use as background images 

python Compositing_code.py  


######################
### data preprocessing: crop into 320*320 random patches  
# This step was intended to shorten the train time by skipping the cropping patch step during training, but training results were not as good as train directly. 
# Therefore, this step was skipped @ ckpt_2 
# Results using this method can be found @ ckpt_1  
### generate input data for train 
#python2
cd ~/Segmentation-Refinement 
emacs Preprocessing_code_for_train.py 

#base_path = '/home/vortex/bonniehu/Adobe_data/processed_data_for_train/' 


######################
### run training
emacs matting_unpool.py 
# change L225:
# 	 * load_data_adobe() # process images during training
# 	 * load_data_adobe_processed() # use already processed patches
cd ~/Segmentation-Refinement
emacs adobe_train.sh 
# set correct directories/hyper parameters
tmux a -t 0
bash adobe_train.sh 
ctrl+b d


######################
### check training results using Tensorboard 
tmux a -t 1
tensorboard --logdir log 
# make sure you have a new log dir for each brand new training 
# tensorboard merges even.logs of same train in same log dir automatically 


######################
### Train Refine Stage
# script matting_refine.py was modified to load first stage, freeze it, train second stage 
# learning rate = 10^-5, no decay
bash adobe_refine.sh 


######################
### Two stage fine tune 
# unfreeze 1st stage, train 1st + 2nd stage together
# learning rate = 0.00001

TBD


######################
### test 
## script test.py
# 	 * (still work on it) modified to cut image patches then puzzle patch results together
#	 * change ckpt dir to switch models
## usage
# Two methods:
# method = 1: keep original image size, crop it into 320*320 patches, run on each patch, then puzzle patches together
# method = 2: resize any input to 320*320, run on resized image, resize to original size

bash adobe_test_one_image.sh #change image by editing script


