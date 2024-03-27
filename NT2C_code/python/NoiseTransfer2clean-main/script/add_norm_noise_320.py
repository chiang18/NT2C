# -*- coding:utf-8 -*
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




def func(p,x,y,noise_mean):
    k1,k2,b=p
    return k1*x+k2*(y-noise_mean)+b

def error(p,x,y,noise_mean,z):
    #print(s)
    return func(p,x,y,noise_mean)-z 



def fit_noise(clean,noise,noisy):
    #print(clean.shape)
    #print(noise.shape)
    #print(noisy.shape)
    clean = np.squeeze(clean)
    noise = np.squeeze(noise)
    noisy = np.squeeze(noisy)

    #noise = noise.reshape((1, 160, 160))
    #noise=noise[0,:,:]
    Xi=np.array([i for j in clean for i in j],dtype=np.float64)
    Yi=np.array([i for j in noise for i in j],dtype=np.float64)
    Zi=np.array([i for j in noisy for i in j],dtype=np.float64)
    noise_mean=np.mean(Yi)
    p0=[100,100,20]
    #print('clean',Xi.shape)
    #print('gan noise',Yi.shape)
    #print('noisy',Zi.shape)
    #print(noise_mean.shape)
    Para=leastsq(error,p0,args=(Xi,Yi,noise_mean,Zi))
    k1,k2,b=Para[0]

    return k1,k2,b



    
clean_dir=sys.argv[1]
noise_dir=sys.argv[2]
noisy_dir=sys.argv[3]
noisy_gen_dir=sys.argv[4]

clean_list =sorted(os.listdir(clean_dir))
noise_list=os.listdir(noise_dir)
noisy_list=sorted(os.listdir(noisy_dir))
noise_len=len(noise_list)
a=len(clean_list)
i=1
#output_zip = os.path.join(noisy_gen_dir, '256_syn.zip')
#with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_LZMA) as zipf:  #ZIP_DEFLATED
# for clean_img in clean_list:
#     rand1=random.randint(0,noise_len-1)
#     #clean_path=clean_dir+clean_img
#     #noisy_path=noisy_dir+clean_img
#     #noise_path=noise_dir+noise_list[rand1]
#     clean_path=os.path.join(clean_dir,clean_img)
#     noisy_path=os.path.join(noisy_dir,clean_img)
#     noise_path=os.path.join(noise_dir,noise_list[rand1])

#     clean=mrcfile.open(clean_path)
#     noisy=mrcfile.open(noisy_path)
#     noise=mrcfile.open(noise_path)
#     clean=clean.data
#     noisy=noisy.data
#     noise=noise.data

#     clean=(clean-np.min(clean))/(np.max(clean)-np.min(clean))*255.0
#     noisy=(noisy-np.min(noisy))/(np.max(noisy)-np.min(noisy))*255.0
#     noise=(noise-np.min(noise))/(np.max(noise)-np.min(noise))*255.0

#     #k1,k2,b=fit_noise(clean,noise,noisy)
#     #print('{}/{},k1,k2,b: {}, {}, {}'.format(i,a,k1,k2,b))
#     print('{}/{}'.format(i,a))
#     i=i+1
#     noisy_gen=k1*clean+k2*(noise-np.mean(noise))+b+(noise-np.mean(noise))

#     #noisy_gen=1*clean+noise
    
#     noisy_gen=(noisy_gen-np.min(noisy_gen))/(np.max(noisy_gen)-np.min(noisy_gen))
    
#     #print('noisy_gen',noisy_gen.max(),noisy_gen.min())
#     #noisy_gen = noisy_gen.reshape((640,640))
#     #noisy_gen = noisy_gen.reshape((320,320))
#     #noisy_gen = noisy_gen.reshape((256,256))
#     fit_noisy_out=mrcfile.new(noisy_gen_dir+clean_img,overwrite=True)
#     fit_noisy_out.set_data(noisy_gen.astype(np.float32))
    
#     mrc_out = mrcfile.new(clean_img, overwrite=True)
#     mrc_out.set_data(noisy_gen.astype(np.float32))
#     mrc_out.close()
#     zipf.write(clean_img)
#     os.remove(clean_img)

def add_noise(clean_img):
    rand1=random.randint(0,noise_len-1)
    #clean_path=clean_dir+clean_img
    #noisy_path=noisy_dir+clean_img
    #noise_path=noise_dir+noise_list[rand1]
    clean_path=os.path.join(clean_dir,clean_img)
    noisy_path=os.path.join(noisy_dir,clean_img)
    noise_path=os.path.join(noise_dir,noise_list[rand1])

    clean=mrcfile.open(clean_path)
    noisy=mrcfile.open(noisy_path)
    noise=mrcfile.open(noise_path)
    clean=clean.data
    noisy=noisy.data
    noise=noise.data

    clean=(clean-np.min(clean))/(np.max(clean)-np.min(clean))*255.0
    noisy=(noisy-np.min(noisy))/(np.max(noisy)-np.min(noisy))*255.0
    noise=(noise-np.min(noise))/(np.max(noise)-np.min(noise))*255.0
#     print('{}/{}'.format(i,a))
#     i=i+1

    k1,k2,b=fit_noise(clean,noise,noisy)
    print('{}/{},k1,k2,b: {}, {}, {}'.format(1,1,k1,k2,b))
    noisy_gen=k1*clean+k2*(noise-np.mean(noise))+b+(noise-np.mean(noise))

    #noisy_gen=0.5*clean+noise
    
    noisy_gen=(noisy_gen-np.min(noisy_gen))/(np.max(noisy_gen)-np.min(noisy_gen))
    
    #print('noisy_gen',noisy_gen.max(),noisy_gen.min())
    #noisy_gen = noisy_gen.reshape((640,640))
    #noisy_gen = noisy_gen.reshape((320,320))
    #noisy_gen = noisy_gen.reshape((256,256))
    fit_noisy_out=mrcfile.new(noisy_gen_dir+clean_img,overwrite=True)
    fit_noisy_out.set_data(noisy_gen.astype(np.float32))

pool = multiprocessing.Pool(processes=12)
pool.map(add_noise, clean_list)

# 关闭进程池
pool.close()
pool.join()

