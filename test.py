'''
CUDA_VISIBLE_DEVICES=0 python test.py
'''
###
# 2 ways to run test
# method = 1: keep original image size, crop it into 320*320 patches, run on each patch, then puzzle patches together
# method = 2: resize any input to 320*320, run on resized image, resize to original size

import tensorflow as tf
import numpy as np
import os
from scipy import misc
#from util import generate_trimap
import argparse
import sys
from util import *

g_mean = np.array(([126.88,120.24,112.19])).reshape([1,1,3])
crop_width = 320
crop_height = 320
train_batch_size = 4
sample_patch_size = np.array([320, 480, 640, 800])

def main(args):
	
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = args.gpu_fraction)
	with tf.Session(config=tf.ConfigProto(gpu_options = gpu_options)) as sess:
		#saver = tf.train.import_meta_graph('/home/vortex/bonniehu/Segmentation-Refinement/meta/my-model.meta')
                
                saver = tf.train.import_meta_graph(args.meta)
                print 'Restoring ckpt ', args.ckpt
		saver.restore(sess,tf.train.latest_checkpoint(args.ckpt))
		#train_batch = tf.get_collection('train_batch')[0]
                #is_training = tf.get_collection('is_training')[0]
                train_batch = tf.get_collection('train_batch')[0]
                pred_mattes = tf.get_collection('pred_mattes')[0]
                alpha_loss = tf.get_collection('alpha_loss')[0]
		
                '''
                # read origin images 
                rgb = misc.imread(args.rgb, mode='RGB')
                rgb = rgb.astype(np.float32)-g_mean
		trimap = misc.imread(args.alpha,'L')
		origin_shape = trimap.shape # height,width
                
                print args.origin
                if args.origin == 1:
                        # resize image to nearest 320*320 grids 
                        new_shape = [int(round(origin_shape[i]/320.0)*320) for i in [0,1]]
                elif args.origin == 0:
                        print 'Resizing to 320 by 320'
                        new_shape = [320, 320]
                else:
                        # random crop patch like what training does
                        print 'Crop patch'
                        new_shape = [320, 320]
                        # copy from: util.load_single_image_adobe(alpha_path, FG_path, BG_path, RGB_path)
                        crop_size = np.random.choice(sample_patch_size)
                        crop_center = crop_patch(trimap[:,:], crop_size, 'adobe')
                        if crop_center is not None:
                                row_start = crop_center[0] - crop_size / 2 + 1
                                row_end = crop_center[0] + crop_size / 2 - 1
                                col_start = crop_center[1] - crop_size / 2 + 1
                                col_end = crop_center[1] + crop_size / 2 - 1
                                trimap = trimap[row_start:row_end, col_start:col_end]
                                rgb = rgb[row_start:row_end, col_start:col_end, :]
                rgb = misc.imresize(rgb.astype(np.uint8),[new_shape[0],new_shape[1],3]).astype(np.float32)
                trimap = misc.imresize(trimap.astype(np.uint8),new_shape, interp = 'nearest').astype(np.float32)
                misc.imsave('./rgb.png',np.squeeze(rgb))
                misc.imsave('./trimap.png',np.squeeze(trimap))
                # crop by grid 
                rgbcrops = []
                trimapcrops = []

                for h in range(new_shape[0]/320):
                        for w in range(new_shape[1]/320):
                                row_start = h * 320
                                row_end = (h + 1) * 320
                                col_start = w * 320
                                col_end = (w + 1) * 320
                                rgbcrops.append(rgb[row_start:row_end, col_start:col_end, :]) 
                                trimapcrops.append(trimap[row_start:row_end, col_start:col_end])
                alphacrops = []
                for rgb, trimap in zip(rgbcrops, trimapcrops):
                        rgb = np.expand_dims(rgb,0)
                        trimap = np.expand_dims(np.expand_dims(trimap.astype(np.float32),2),0)
                        print(np.squeeze(rgb).shape, np.squeeze(trimap).shape)
                        batch = np.concatenate([trimap, trimap, rgb, rgb, rgb, rgb], axis=3)
                        batch = np.concatenate([batch for i in range(batch_size)], axis=0)
                        #feed_dict = {train_batch:batch, is_training:False}
                        feed_dict = {train_batch:batch}
                        loss, pred_alpha = sess.run([total_loss,pred_mattes],feed_dict = feed_dict)
                        alphacrops.append(pred_alpha)
                        print "Total loss: ", loss
                final_alpha = None
                final_trimap = None
                for h in range(new_shape[0]/320):
                        row_piece = None
                        row_piece2 = None
                        for w in range(new_shape[1]/320):
                                if row_piece is None:
                                        row_piece = alphacrops[h*int(new_shape[1]/320)+w]
                                        row_piece2 = trimapcrops[h*int(new_shape[1]/320)+w]
                                else:
                                        row_piece = np.concatenate((row_piece, alphacrops[h*int(new_shape[1]/320)+w]), axis=2)
                                        row_piece2 = np.concatenate((row_piece2, trimapcrops[h*int(new_shape[1]/320)+w]), axis=1)
                                print row_piece.shape
                        if final_alpha is None:
                                final_alpha = row_piece
                                final_trimap = row_piece2
                        else:
                                final_alpha = np.concatenate((final_alpha, row_piece), axis=1)
                                final_trimap = np.concatenate((final_trimap, row_piece2), axis=0)
                        print final_alpha.shape, final_trimap.shape
		final_alpha = misc.imresize(np.squeeze(final_alpha[-1]),origin_shape)
                final_trimap = misc.imresize(np.squeeze(final_trimap),origin_shape)
                misc.imsave('./alpha.png',final_alpha)
                misc.imsave('./trimap.png',final_trimap)
                '''
                print('test on validation data...')
                val_loss = []
                validation_dir = '/home/vortex/bonniehu/Mask_data/'
                #validation_dir = '/home/vortex/bonniehu/Benchmark_data/Test_set/high_resolution/'
                #validation_dir = '/home/vortex/bonniehu/Benchmark_data/Test_set/low_resolution/'
                #validation_dir = '/home/vortex/bonniehu/Adobe_data/Test_set/'
                #validation_dir = '/home/vortex/bonniehu/Val_data/'
                #validation_dir = '/home/vortex/bonniehu/Mask_data/'
                t_paths_alpha, t_paths_trimap, t_paths_FG, t_paths_BG, t_paths_RGB = load_path_adobe(validation_dir+'alpha', validation_dir+'trimaps', validation_dir+'FG', validation_dir+'BG', validation_dir+'RGB', 'no_loss')
                num_batch = len(t_paths_trimap)/train_batch_size

                for im_index in range(len(t_paths_trimap)):
                        # crop semi-random 
                        #im_index = 15
                        trimap = misc.imread(t_paths_trimap[im_index],'L')
                        print trimap[0], trimap.shape
                        trimap = np.expand_dims(trimap,2)
                        misc.imsave('./result_maskrcnn/trimap_{}.png'.format(im_index), np.squeeze(trimap))
                        rgb = misc.imread(t_paths_RGB[im_index], mode='RGB')
                        print rgb.shape
                        rgb = rgb.astype(np.float32)-g_mean
                        # initialize alpha as equal to trimap 
                        alpha = trimap/255 
                        patch_borders, trimap_return = crop_patch_whole_image(trimap.copy())
                        # make sure number of patches can be exactly divided into train batch size
                        if len(patch_borders) % train_batch_size != 0: 
                                for i in range(train_batch_size - len(patch_borders) % train_batch_size): 
                                        patch_borders.append(patch_borders[-1])
                        num_batch = len(patch_borders)/train_batch_size
                        for i in range(num_batch):
                                begin_i = i*train_batch_size
                                end_i = (i+1)*train_batch_size
                                test_batch = []
                                for border in patch_borders[begin_i: end_i]:
                                        patch = trimap[border[0]:border[1], border[2]:border[3]]
                                        rgb_patch = rgb[border[0]:border[1], border[2]:border[3]]
                                        #misc.imsave('./patch/patch{}.png'.format(i), np.squeeze(patch))
                                        print patch.shape, rgb_patch.shape, border
                                        batch_i = np.concatenate([patch, patch, rgb_patch, rgb_patch, rgb_patch, rgb_patch],2)
                                        test_batch.append(batch_i)
                        
                                test_batch = np.stack(test_batch)
                                feed = {train_batch:test_batch}
                                #run matting with trimap and rgb at train_batch_size
                                test_mattes = sess.run(pred_mattes,feed_dict = feed)
                                test_mattes.reshape([train_batch_size,image_height,image_width,1])
                                for j in range(train_batch_size):
                                        # paste patch results to alpha
                                        # TODO: TAKE MEAN for multiple values on one pixel
                                        border = patch_borders[begin_i: end_i][j]
                                        alpha[border[0]:border[1], border[2]:border[3]] = test_mattes[j]
                        misc.imsave('./result_maskrcnn/alpha_{}.png'.format(im_index), np.squeeze(alpha))
                exit()
                
                for i in range(num_batch):
                        begin_i = i*train_batch_size
                        end_i = (i+1)*train_batch_size
                        # test_batch = load_data_adobe(batch_alpha_paths=t_paths_alpha[begin_i:end_i], batch_FG_paths=t_paths_FG[begin_i:end_i], batch_BG_paths=t_paths_BG[begin_i:end_i], batch_RGB_paths=t_paths_RGB[begin_i:end_i], no_alpha=True, batch_trimap_paths=t_paths_trimap[begin_i:end_i])
                        test_batch = load_data_adobe(batch_RGB_paths=t_paths_RGB[begin_i:end_i], loss='no_loss', batch_trimap_paths=t_paths_trimap[begin_i:end_i])#,batch_alpha_paths=t_paths_alpha[begin_i:end_i])
                        feed = {train_batch:test_batch}
                        aloss, val_mattes = sess.run([alpha_loss, pred_mattes],feed_dict = feed)
                        val_mattes.reshape([train_batch_size,image_height,image_width,1])
                        val_loss.append(aloss)
                        print 'alpha loss is ', aloss 
                        for j in range(train_batch_size):
                            misc.imsave('./test_out/{}_{}_{}.png'.format(aloss,i,j), np.squeeze(val_mattes[j]))
                val_mean_loss = np.mean(val_loss)
                print 'length of val_loss ', len(val_loss)
                print('average test loss is '+ str(val_mean_loss))
                
def parse_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument('--alpha', type=str,
		help='input alpha')
	parser.add_argument('--rgb', type=str,
		help='input rgb')
	parser.add_argument('--gpu_fraction', type=float,
		help='how much gpu is needed, usually 4G is enough',default = 0.4)
        parser.add_argument('--meta', type=str,
		help='meta graph of model')
        parser.add_argument('--ckpt', type=str,
		help='ckpt dir to load model')
        parser.add_argument('--origin', type=int,
		help='1 to keep original size of test image; other number for not using origin')
	return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

