##Copyright 2017 Adobe Systems Inc.
##
##Licensed under the Apache License, Version 2.0 (the "License");
##you may not use this file except in compliance with the License.
##You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
##Unless required by applicable law or agreed to in writing, software
##distributed under the License is distributed on an "AS IS" BASIS,
##WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##See the License for the specific language governing permissions and
##limitations under the License.


##############################################################
#Set your paths here

#path to provided foreground images
fg_path = '/home/vortex/bonniehu/AdobeData/Other/fg/'

#path to provided alpha mattes
a_path = '/home/vortex/bonniehu/AdobeData/Other/alpha/'

#Path to background images (MSCOCO)
bg_path = '/home/vortex/bonniehu/COCO2017/'

#Path to folder where you want the composited images to go
out_path = '/home/vortex/bonniehu/data/compose/'

out_fg_path = '/home/vortex/bonniehu/data/fg/'

out_bg_path = '/home/vortex/bonniehu/data/bg/'

out_alpha_path = '/home/vortex/bonniehu/data/alpha/'

##############################################################

from PIL import Image
import os 
import math
import time
import numpy as np
from scipy import misc

def composite4(fg, bg, a, w, h):
    
    bbox = fg.getbbox()
    bg = bg.crop((0,0,w,h))
    
    fg_list = fg.load()
    bg_list = bg.load()
    a_list = a.load()
    
    for y in range(h):
        for x in range (w):
            alpha = a_list[x,y] / 255.0
            t = fg_list[x,y][0]
            t2 = bg_list[x,y][0]
            if alpha >= 1:
                r = int(fg_list[x,y][0])
                g = int(fg_list[x,y][1])
                b = int(fg_list[x,y][2])
                bg_list[x,y] = (r, g, b, 255)
            elif alpha > 0:
                r = int(alpha * fg_list[x,y][0] + (1-alpha) * bg_list[x,y][0])
                g = int(alpha * fg_list[x,y][1] + (1-alpha) * bg_list[x,y][1])
                b = int(alpha * fg_list[x,y][2] + (1-alpha) * bg_list[x,y][2])
                bg_list[x,y] = (r, g, b, 255)

    return bg

num_bgs = 10

fg_files = os.listdir(fg_path)
a_files = os.listdir(a_path)
bg_files = os.listdir(bg_path)

bg_iter = iter(bg_files)
for im_name in fg_files:
    
    im = Image.open(fg_path + im_name);
    a = Image.open(a_path + im_name);
    bbox = im.size
    w = bbox[0]
    h = bbox[1]
    
    if im.mode != 'RGB' and im.mode != 'RGBA':
        im = im.convert('RGB')
    
    bcount = 0 
    for i in range(num_bgs):

        bg_name = next(bg_iter, -1)
        if bg_name==-1:
            bg_iter = iter(bg_files)
            bg_name = next(bg_iter)
        print (im_name + '_' + str(bcount))
        #out.save(out_path + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png', "PNG")
        if os.path.exists(out_alpha_path + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png'):
            bcount += 1
            continue
        bg = Image.open(bg_path + bg_name)
        if bg.mode != 'RGB':
            bg = bg.convert('RGB')

        bg_bbox = bg.size
        bw = bg_bbox[0]
        bh = bg_bbox[1]
        wratio = w / bw
        hratio = h / bh
        ratio = wratio if wratio > hratio else hratio 
        if ratio > 1:        
            bg = bg.resize((int(math.ceil(bw*ratio)),int(math.ceil(bh*ratio))), Image.BICUBIC)
        
        bg = bg.crop((0,0,w,h))
        
        fg = np.array(im)
        alpha = np.array(a)
        bng = np.array(bg)
        alpha_3 = np.dstack((alpha,alpha,alpha)) / 255.0
        out = fg * alpha_3 + bng * (1 - alpha_3)
        fg = fg * alpha_3

        #a.save(out_alpha_path + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png', "PNG")
        #bg.save(out_bg_path + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png', "PNG")
        #im.save(out_fg_path + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png', "PNG")
        #out = composite4(im, bg, a, w, h)
        misc.imsave(out_alpha_path + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png', alpha)
        misc.imsave(out_bg_path + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png', bng)
        misc.imsave(out_fg_path + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png', fg)
        misc.imsave(out_path + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png', out)
        bcount += 1



