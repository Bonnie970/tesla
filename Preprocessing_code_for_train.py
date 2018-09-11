#Set your paths here

#Path to folder where you want the composited images to go
out_rgb_path = '/home/vortex/bonniehu/Adobe_data/Training_set/composed_images/'

out_fg_path = '/home/vortex/bonniehu/Adobe_data/Training_set/composed_fg/'

out_bg_path = '/home/vortex/bonniehu/Adobe_data/Training_set/composed_bg/'

out_alpha_path = '/home/vortex/bonniehu/Adobe_data/Training_set/composed_alpha/'

#Output paths
base_path = '/home/vortex/bonniehu/Adobe_data/processed_data_for_train/'
path_index = '1'
out2_alpha_path = base_path + 'alpha_' + path_index + '/'
out2_trimap_path = base_path + 'trimap_' + path_index + '/'
out2_gmean_path = base_path + 'gmean_' + path_index + '/'
out2_fg_path = base_path + 'fg_' + path_index + '/'
out2_bg_path = base_path + 'bg_' + path_index + '/'
out2_rgb_path =  base_path + 'rgb_' + path_index + '/'

image_height = 320 
image_width = 320

##############################################################
from joblib import Parallel, delayed
from PIL import Image
import cv2
import os 
import math
import time
import numpy as np
#from scipy import misc
from util import *

g_mean = np.array(([126.88,120.24,112.19])).reshape([1,1,3])
sample_patch_size = np.array([320, 480, 640])


def mkdir():
    for path in [out2_alpha_path, out2_trimap_path,out2_gmean_path,out2_fg_path,out2_bg_path,out2_rgb_path]: 
        if not os.path.exists(path):
            os.makedirs(path)


def crop_save_1_image(im_name):
    print ('Cropping ', im_name)
    alpha = Image.open(out_alpha_path + im_name)#.convert('L')
    alpha = np.expand_dims(alpha,2)
    trimap = np.copy(alpha)
    trimap = generate_trimap(trimap, alpha)
    rgb = cv2.imread(out_rgb_path + im_name)
    fg = cv2.imread(out_fg_path + im_name)
    bg = cv2.imread(out_bg_path + im_name)
    # random sample 320/480/640/800 patch, then resize to 320*320
    crop_size = np.random.choice(sample_patch_size)
    crop_center = crop_patch(trimap[:,:,0], crop_size, 'adobe')
    if crop_center is not None:
        row_start = crop_center[0] - crop_size / 2 + 1
        row_end = crop_center[0] + crop_size / 2 - 1
        col_start = crop_center[1] - crop_size / 2 + 1
        col_end = crop_center[1] + crop_size / 2 - 1
        alpha = alpha[row_start:row_end, col_start:col_end, :]
        trimap = trimap[row_start:row_end, col_start:col_end, :]
        rgb = rgb[row_start:row_end, col_start:col_end, :]
        fg = fg[row_start:row_end, col_start:col_end, :]
        bg = bg[row_start:row_end, col_start:col_end, :]
    #rgb.resize((image_width, image_height), Image.BICUBIC)
    # save images
    imgs = [alpha, trimap, fg, bg, rgb] #[alpha, trimap, rgb - g_mean, fg, bg, rgb]
    paths = [out2_alpha_path, out2_trimap_path,out2_fg_path,out2_bg_path,out2_rgb_path] #[out2_alpha_path, out2_trimap_path,out2_gmean_path,out2_fg_path,out2_bg_path,out2_rgb_path]
    for i in range(len(imgs)):
        print paths[i] 
        img = imgs[i].copy()
        # squeeze black white alpha matte and trimap 
        if i < 2: 
            if imgs[i].shape[2]==1:
                img = np.squeeze(imgs[i], axis=2)
        # resize to 320*320
        resize = cv2.resize(img, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
        # save
        #misc.imsave(path + im_name[:len(im_name)-4] + '_' + path_index + '.jpg', img)
        cv2.imwrite(paths[i] + im_name[:len(im_name)-4] + '_' + path_index + '.jpg', resize)


def crop_image_1by1():
    files = os.listdir(out_rgb_path)
    print ('Processing ', len(files), ' images...')
    for im_name in files:
        crop_save_1_image(im_name)
        

def crop_image_parallel(n):
    files = os.listdir(out_rgb_path)
    print ('Processing ', len(files), ' images...')
    num_group = int(len(files)/n)
    for i in range(num_group):
        Parallel(n_jobs=n, verbose=1, backend="threading")(map(delayed(crop_save_1_image), files[i*n:(i+1)*n]))
    

if __name__ == "__main__":
    mkdir()
    crop_image_parallel(4)

