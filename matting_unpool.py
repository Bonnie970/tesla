import tensorflow as tf
import gpumemory
import numpy as np
from util import *
import os
from scipy import misc
import timeit
from net import base_net
import random 

flags = tf.app.flags
flags.DEFINE_string('alpha_path', None, 'Path to alpha files')
flags.DEFINE_string('trimap_path', None, 'Path to trimap files')
flags.DEFINE_string('fg_path', None, 'Path to fg files')
flags.DEFINE_string('bg_path', None, 'Path to bg files')
flags.DEFINE_string('rgb_path', None, 'Path to rgb files')
flags.DEFINE_string('model_path', None, 'path to VGG weights')
flags.DEFINE_string('log_dir', None, 'Path to save logs')
flags.DEFINE_string('save_ckpt_path', None, 'Path to save ckpt files')
flags.DEFINE_string('fine_tune_ckpt_path', None, 'Path to pretrained ckpt files')
flags.DEFINE_string('save_meta_path', None, 'Path to save meta data')
flags.DEFINE_string('dataset_name', None, 'dataset name, "Adobe", "DAVIS"')
flags.DEFINE_integer('image_height', 320, 'input image height')
flags.DEFINE_integer('image_width', 320, 'input image width')
flags.DEFINE_integer('max_epochs', 500, 'max epochs to run' )
flags.DEFINE_integer('batch_size', 8, 'batch_size for training')
flags.DEFINE_integer('save_log_steps', 50, 'save log after steps')
flags.DEFINE_integer('save_ckpt_steps', 5000, 'save ckpt after steps')
flags.DEFINE_float('learning_rate', 0.0004, 'initial learning rate')
flags.DEFINE_float('learning_rate_decay', 0.95, 'learning rate decay factor')
flags.DEFINE_float('learning_rate_decay_steps', 100, 'learning rate decay after epochs')
flags.DEFINE_boolean('restore_from_ckpt', 'False', 'Whether restore weights form ckpt file')
flags.DEFINE_boolean('use_focal_loss', 'False', 'Whether use focal loss')

FLAGS = flags.FLAGS

def main(_):
    image_height = FLAGS.image_height
    image_width = FLAGS.image_width
    train_batch_size = FLAGS.batch_size
    max_epochs = FLAGS.max_epochs
    hard_mode = False

    pretrained_model = FLAGS.restore_from_ckpt

    #pretrained_vgg_model_path
    model_path = FLAGS.model_path
    log_dir = FLAGS.log_dir

    dataset_alpha = FLAGS.alpha_path
    dataset_trimap = FLAGS.trimap_path
    dataset_RGB = FLAGS.rgb_path
    dataset_fg = FLAGS.fg_path
    dataset_bg = FLAGS.bg_path
    if FLAGS.dataset_name == 'DAVIS':
        #paths_alpha,paths_trimap,paths_RGB = load_path(dataset_alpha,dataset_trimap,dataset_RGB)
        paths_alpha, paths_trimap, paths_FG, paths_BG, paths_RGB = load_path_DAVIS(dataset_alpha, dataset_trimap,dataset_fg, dataset_bg, dataset_RGB)
    else:
        paths_alpha, paths_trimap, paths_FG, paths_BG, paths_RGB = load_path_adobe(dataset_alpha, dataset_trimap, dataset_fg, dataset_bg, dataset_RGB)
    range_size = len(paths_alpha)
    print('range_size is %d' % range_size)
    #range_size/batch_size has to be int
    batchs_per_epoch = int(range_size/train_batch_size) 

    index_queue = tf.train.range_input_producer(range_size, num_epochs=None,shuffle=True, seed=None, capacity=32)
    index_dequeue_op = index_queue.dequeue_many(train_batch_size, 'index_dequeue')

    train_batch = tf.placeholder(tf.float32, shape=(train_batch_size, image_height, image_width, 14))
    print 'train batch size is ', train_batch_size
    tf.add_to_collection('train_batch', train_batch)
    
    #is_training = tf.Variable(0,name='is_training',trainable=False)
    #tf.add_to_collection("is_training", is_training)
    
    images = tf.map_fn(lambda img: image_preprocessing(img, is_training=False), train_batch)
    # images = tf.map_fn(lambda img: image_preprocessing(img, is_training=True), train_batch)
    
    b_GTmatte, b_trimap, b_RGB, b_GTFG, b_GTBG, raw_RGBs = tf.split(images, [1, 1, 3, 3, 3, 3], 3)

    #tf.add_to_collection('trimap', b_trimap)
    #tf.add_to_collection('raw_RGBs', raw_RGBs)

    tf.summary.image('GT_matte_batch',b_GTmatte,max_outputs = 4)
    tf.summary.image('trimap',b_trimap,max_outputs = 4)
    tf.summary.image('raw_RGBs',raw_RGBs,max_outputs = 4)

    b_input = tf.concat([b_RGB,b_trimap],3)
    
    pred_mattes, en_parameters = base_net(b_input, trainable=True, training=True)

    tf.add_to_collection("pred_mattes", pred_mattes)

    
    if FLAGS.dataset_name == 'DAVIS':
	if FLAGS.use_focal_loss:
	    print 'using focal loss'
            wl = tf.where(tf.logical_and(tf.greater(b_trimap,5), tf.less(b_trimap, 250)), tf.fill([train_batch_size,image_width,image_height,1],1.), tf.fill([train_batch_size,image_width,image_height,1], 0.1))
        else:
            wl = tf.where(tf.logical_and(tf.greater(b_trimap,5), tf.less(b_trimap, 250)), tf.fill([train_batch_size,image_width,image_height,1],1.), tf.fill([train_batch_size,image_width,image_height,1], 0.1))
    else:
	if FLAGS.use_focal_loss:
	    print 'using focal loss'
            wl = tf.where(tf.equal(b_trimap,128), tf.fill([train_batch_size,image_width,image_height,1],1.), tf.fill([train_batch_size,image_width,image_height,1], 0.))
        else:
            # known region 0. -> 0.5
            wl = tf.where(tf.equal(b_trimap,128), tf.fill([train_batch_size,image_width,image_height,1],1.), tf.fill([train_batch_size,image_width,image_height,1], 0.25))
    tf.summary.image('pred_mattes',pred_mattes,max_outputs = 4)
    tf.summary.image('wl',wl,max_outputs = 4)
    #alpha_diff = tf.sqrt(tf.square(pred_mattes/255.0 - b_GTmatte/255.0,)  + 1e-12)
    if FLAGS.use_focal_loss:
   	alpha_diff = tf.square(pred_mattes - b_GTmatte/255.0,) + 1e-12
    else:
    	alpha_diff = tf.sqrt(tf.square(pred_mattes - b_GTmatte/255.0,) + 1e-12)

    p_RGB = []
    pred_mattes.set_shape([train_batch_size,image_height,image_width,1])
    b_GTBG.set_shape([train_batch_size,image_height,image_width,3])
    b_GTFG.set_shape([train_batch_size,image_height,image_width,3])
    raw_RGBs.set_shape([train_batch_size,image_height,image_width,3])
    b_GTmatte.set_shape([train_batch_size,image_height,image_width,1])

    # pred_final =  tf.where(tf.equal(b_trimap,128), pred_mattes, b_trimap/255.0)
    # tf.summary.image('pred_final',pred_final,max_outputs = 5)
    
    l_matte = tf.unstack(pred_mattes)
    BG = tf.unstack(b_GTBG)
    FG = tf.unstack(b_GTFG)

    for i in range(train_batch_size):
        #p_RGB.append(BG[i] - FG[i])
        #p_RGB.append((tf.ones_like(l_matte[i], dtype=tf.float32) - l_matte[i] / 255.0) * BG[i])
        #p_RGB.append(l_matte[i] / 255.0 * FG[i] + (tf.constant(1.) - l_matte[i] / 255.0) * BG[i])
        p_RGB.append(l_matte[i] * FG[i] +  (tf.constant(1.) - l_matte[i]) * BG[i])
        #p_RGB.append(l_matte[i] / 255.0 * FG[i] + (tf.constant(1.) - l_matte[i] / 255.0) * BG[i])
    pred_RGB = tf.stack(p_RGB)
    tf.summary.image('pred_RGB', pred_RGB, max_outputs = 4)
    tf.summary.image('GTFG', b_GTFG, max_outputs = 4)
    tf.summary.image('GTBG', b_GTBG, max_outputs = 4)
    #c_diff = tf.sqrt(tf.square(pred_RGB/255.0 - raw_RGBs/255.0) + 1e-12)
    # changed 201709
    # TODO figure out how to deal with this loss
    #c_diff = tf.sqrt(tf.square(pred_RGB/255.0 - raw_RGBs/255.0) + 1e-12)
    if FLAGS.use_focal_loss:
    	c_diff = tf.square(pred_RGB/255.0 - raw_RGBs/255.0) + 1e-12
    else:
    	c_diff = tf.sqrt(tf.square(pred_RGB/255.0 - raw_RGBs/255.0) + 1e-12)

    unknown_region_size = tf.reduce_sum(wl)
    alpha_loss = tf.reduce_sum(alpha_diff*wl) / (unknown_region_size) #tf.reduce_sum(wl) / 2.
    comp_loss = tf.reduce_sum(c_diff*wl) / (unknown_region_size) #tf.reduce_sum(wl) / 2.
    #alpha_loss = tf.reduce_sum(alpha_diff * wl)/(tf.reduce_sum(wl))
    #comp_loss = tf.reduce_sum(c_diff * wl)/(tf.reduce_sum(wl))

    # tf.summary.image('alpha_diff',alpha_diff * wl_alpha,max_outputs = 5)
    # tf.summary.image('c_diff',c_diff * wl_RGB,max_outputs = 5)

    tf.summary.scalar('alpha_loss',alpha_loss)
    tf.summary.scalar('comp_loss',comp_loss)

    total_loss = (alpha_loss + comp_loss) * 0.5
    tf.summary.scalar('total_loss',total_loss)

    tf.add_to_collection('total_loss',total_loss)
    tf.add_to_collection('alpha_loss',alpha_loss)
    tf.add_to_collection('comp_loss',comp_loss)
    
    global_step = tf.Variable(0,name='global_step',trainable=False)

    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                                global_step,
                                                FLAGS.learning_rate_decay_steps,
                                                FLAGS.learning_rate_decay,
                                                staircase=True,
                                                name='exponential_decay_learning_rate')
    tf.summary.scalar('learning_rate',learning_rate)
    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss,global_step = global_step)
    

    #saver = tf.train.Saver(tf.trainable_variables() , max_to_keep = 10)
    saver = tf.train.Saver(max_to_keep = 10)

    coord = tf.train.Coordinator()
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

    gpu_options = tf.GPUOptions()#(per_process_gpu_memory_fraction = 0.2)
    with tf.Session(config=tf.ConfigProto(gpu_options = gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(coord=coord,sess=sess)
        batch_num = 0
        epoch_num = 0
        #initialize all parameters in vgg16
        if not pretrained_model:
            weights = np.load(model_path)
            keys = sorted(weights.keys())
            for i, k in enumerate(keys):
                if i == 26:
                    break
                if k == 'conv1_1_W':  
                    sess.run(en_parameters[i].assign(np.concatenate([weights[k],np.zeros([3,3,1,64])],axis = 2)))
                else:
                    sess.run(en_parameters[i].assign(weights[k]))
            print('finish loading vgg16 model')
            print os.system('nvidia-smi')
        else:
            print FLAGS.fine_tune_ckpt_path is None
            if FLAGS.fine_tune_ckpt_path is None:
                print('Restoring last ckpt...')
                saver.restore(sess,tf.train.latest_checkpoint(FLAGS.save_ckpt_path))
            else:
                print('Restoring pretrained model...')
                saver.restore(sess,tf.train.latest_checkpoint(FLAGS.fine_tune_ckpt_path))
                global_step.assign(0).eval()
            print('Restoring finished')
        sess.graph.finalize()
        epoch_num = global_step.eval() * train_batch_size // range_size
        while epoch_num < max_epochs:
            while batch_num < batchs_per_epoch:
                batch_index = sess.run(index_dequeue_op)
                total_start = timeit.default_timer()
                if FLAGS.dataset_name == 'DAVIS':
                    batch_alpha_paths = paths_alpha[batch_index]
                    batch_trimap_paths = paths_trimap[batch_index]
                    batch_FG_paths = paths_FG[batch_index]
                    batch_BG_paths = paths_BG[batch_index]
                    batch_RGB_paths = paths_RGB[batch_index]
                    images_batch = load_data_DAVIS(batch_alpha_paths,batch_trimap_paths,batch_FG_paths,batch_BG_paths,batch_RGB_paths)
                else:
                    batch_alpha_paths = paths_alpha[batch_index]
                    batch_trimap_paths = paths_trimap[batch_index]
                    batch_FG_paths = paths_FG[batch_index]
                    batch_BG_paths = paths_BG[batch_index]
                    batch_RGB_paths = paths_RGB[batch_index]
                    images_batch = load_data_adobe(batch_alpha_paths, batch_FG_paths,batch_BG_paths,batch_RGB_paths)
                feed = {train_batch:images_batch}
                #print 'after feed data'
                #print os.system('nvidia-smi')
                train_start = timeit.default_timer()
                _,loss,summary_str,step= sess.run([train_op,total_loss,summary_op,global_step],feed_dict = feed)
                #print os.system('nvidia-smi')
                train_end = timeit.default_timer()
                if step%FLAGS.save_ckpt_steps == 2:
                    saver.export_meta_graph(FLAGS.save_meta_path)
                    print('saving model......')
                    saver.save(sess,FLAGS.save_ckpt_path + '/model.ckpt',global_step = global_step, write_meta_graph = True)
                if step%FLAGS.save_ckpt_steps == 2:
                    print('test on validation data...')
                    val_loss = []
                    validation_dir = '/home/vortex/bonniehu/Val_data/'#'/home/vortex/bonniehu/Adobe_data/Test_set/'
                    t_paths_alpha, t_paths_trimap, t_paths_FG, t_paths_BG, t_paths_RGB = load_path_adobe(validation_dir+'alpha', validation_dir+'trimaps',validation_dir+'FG', validation_dir+'BG', validation_dir+'RGB')
                    validation_summary = tf.Summary()
                    if len(t_paths_alpha) == 0: 
                        num_batch = len(t_paths_trimap)/train_batch_size
                    else: 
                        num_batch = len(t_paths_alpha)/train_batch_size
                    for i in range(num_batch):
                        begin_i = i*train_batch_size
                        end_i = (i+1)*train_batch_size
                        test_batch = load_data_adobe(batch_alpha_paths=t_paths_alpha[begin_i:end_i], batch_FG_paths=t_paths_FG[begin_i:end_i], batch_BG_paths=t_paths_BG[begin_i:end_i], batch_RGB_paths=t_paths_RGB[begin_i:end_i])
                        #test_batch = load_data_adobe(batch_RGB_paths=t_paths_RGB[begin_i:end_i], no_alpha=True, batch_trimap_paths=t_paths_trimap[begin_i:end_i])
                        feed = {train_batch:test_batch}
                        loss, val_mattes = sess.run([total_loss, pred_mattes],feed_dict = feed)
                        val_mattes.reshape([train_batch_size,image_height,image_width,1])
                        val_loss.append(loss)
                        for j in range(train_batch_size):
                            misc.imsave('./val_out/{}_{}_{}.png'.format(loss,i,j), np.squeeze(val_mattes[j]))
                    val_mean_loss = np.mean(val_loss)
                    print 'length of val_loss ', len(val_loss)
                    print('validation loss is '+ str(val_mean_loss))
                    validation_summary.value.add(tag='validation_loss',simple_value = val_mean_loss)
                    summary_writer.add_summary(validation_summary,step)
                
                if step%FLAGS.save_log_steps == 0:
                    summary_writer.add_summary(summary_str,global_step = step)
                batch_num += 1
                total_end = timeit.default_timer()
                print('epoch: %d, global_step: %d, loss is %f, batch_train_time: %f, batch_total_time: %f' \
                        %(epoch_num, step, loss, train_end - train_start, total_end - total_start))
            batch_num = 0
            epoch_num += 1

if __name__ == '__main__':
    tf.app.run()
