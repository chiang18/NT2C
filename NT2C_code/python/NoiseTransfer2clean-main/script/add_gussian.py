 #-*- coding:utf-8 -*
import numpy as np
from scipy.optimize import leastsq
import sys,os
import cv2
import operator
import mrcfile
import random
import mrcfile
import zipfile
import numpy as np
import gzip
from functools import reduce
import multiprocessing
import random



    
clean_dir=sys.argv[1]
noise_dir=sys.argv[2]

clean_list =sorted(os.listdir(clean_dir))[:]
a=len(clean_list)
i=1
#print(clean_list)

def add_gaussian_noise(clean_list):  
    clean_path=os.path.join(clean_dir,clean_list)
    em=mrcfile.open(clean_path,permissive=True)
    img=em.data
    img_var=img[0,:,:] #aspire (1,4096,4096)
    #print(img.shape)
    snr = 0.1
    original_std = np.std(img_var)
    noise = np.random.normal(0, original_std / snr, img_var.shape)
    noise = noise[np.newaxis,:, :]
    noisy_image = img + noise
    noisy_image = noisy_image.astype(np.float32)
    with mrcfile.new(noise_dir+clean_list,overwrite=True) as mrc:
        mrc.set_data(noisy_image)
    

pool = multiprocessing.Pool(processes=36)
pool.map(add_gaussian_noise, clean_list)

# 关闭进程池
pool.close()
pool.join()