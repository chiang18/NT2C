#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from matplotlib import pyplot as plt
import mrcfile
import mahotas
import sys,os
import matplotlib.pyplot as plt
import pywt
import multiprocessing


def is_noise(arr,x0,x1,y0,y1,a):
    if (x0+x1)%2==0:
        x1=x1
    else:
        x1=x1-1
    if (y0+y1)%2==0:
        y1=y1
    else:
        y1=y1-1

    x_center=int(box_size*0.5)
    y_center=int(box_size*0.5)
    #print("x0:{}|x1:{}|y0:{}|y1:{}".format(x0,x1,y0,y1))
    crop_1=arr[:y_center,:x_center]
    crop_2=arr[:y_center,x_center:]
    crop_3=arr[y_center:,:x_center]
    crop_4=arr[y_center:,x_center:]

    #std_global=np.std(arr)
    std_global=a*10 #95
    std_1=np.std(crop_1)
    std_2=np.std(crop_2)
    std_3=np.std(crop_3)
    std_4=np.std(crop_4)
    #print('std_global: {} | std_1: {} | std_2: {} | std_3: {} | std_4: {} '.format(std_global,std_1,std_2,std_3,std_4))
    #print('std:',max(std_1, std_2, std_3, std_4))
    
    isnoise=True
    if std_1>0.1*std_global:
        isnoise=False
        return isnoise,None
    if std_2>0.1*std_global:
        isnoise=False
        return isnoise,None
    if std_3>0.1*std_global:
        isnoise=False
        return isnoise,None
    if std_4>0.1*std_global:
        isnoise=False
        return isnoise,None
    print('std:',max(std_1, std_2, std_3, std_4))
    return isnoise,max(std_1, std_2, std_3, std_4)

denoised_dir=sys.argv[1]  #micrographs
raw_dir=sys.argv[2]  #raw micrographs
noise_dir=sys.argv[3]  #noise patches
draw_dir=sys.argv[4] #draw noise 


box_size=256
par_size=200
step=int(0.02*box_size)
#print('step:  {}'.format(step))
noise_patch_num=0
ii=1
input_list=sorted(os.listdir(denoised_dir))[:]

def process_image(input_file):
    denoised_img=cv2.imread(os.path.join(denoised_dir,input_file), cv2.IMREAD_GRAYSCALE)
    denoised_img=((denoised_img-np.min(denoised_img))/(np.max(denoised_img)-np.min(denoised_img))*255)
    # mrc
    file_name, file_ext = os.path.splitext(input_file)
    input_file_mrc = file_name + ".mrc"
    raw_img=mrcfile.open(os.path.join(raw_dir,input_file_mrc),permissive=True)
    raw_img=raw_img.data
    #jpg
    #raw_img=cv2.imread(os.path.join(raw_dir,input_file), cv2.IMREAD_GRAYSCALE)
    
    raw_img=((raw_img-np.min(raw_img))/(np.max(raw_img)-np.min(raw_img))*255)
    
    w,h=denoised_img.shape
    noise_patch_path=os.path.join(noise_dir,input_file_mrc)
    draw_noise_path=os.path.join(draw_dir,input_file)
    
    pa=[]
    for x in range(0,w,box_size):
        for y in range(0,h,box_size):
            denoised_arr=denoised_img[x:x+box_size,y:y+box_size]
            pa.append(np.std(denoised_arr))
    pa=sorted(pa)
    a=pa[int(len(pa)*0.05)]
    noise_patch_n=0
    for x in range(0,w,step):
        for y in range(0,h,step):
            if x+box_size>w or y+box_size>h:
                continue
            denoised_arr=denoised_img[x:x+box_size,y:y+box_size]
            raw_arr=raw_img[x:x+box_size,y:y+box_size]
            noise=is_noise(denoised_arr,x,x+box_size,y,y+box_size,a)
            if noise[0]:  
                #cv2.imwrite(noise_patch_path[:-4]+'_'+str(noise_patch_n)+'.jpg', raw_arr);
                with mrcfile.new(noise_patch_path[:-4]+'_'+str(noise_patch_n)+'.mrc',overwrite=True) as noise_patch:
                     noise_patch.set_data(raw_arr)
                cv2.rectangle(denoised_img, (y,x), (y+box_size,x+box_size), (255, 0, 0), 2)
                noise_patch_n+=1
    cv2.imwrite(draw_noise_path, denoised_img, [cv2.IMWRITE_JPEG_QUALITY, 75]);
     

pool = multiprocessing.Pool(processes=24)
#print(input_list)
pool.map(process_image, input_list[:])

# 关闭进程池
pool.close()
pool.join()

