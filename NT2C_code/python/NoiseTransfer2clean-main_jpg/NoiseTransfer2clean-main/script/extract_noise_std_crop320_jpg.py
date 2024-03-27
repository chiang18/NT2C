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


def is_noise(arr,x0,x1,y0,y1):
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
#     crop_1=(crop_1-np.min(crop_1))/(np.max(crop_1)-np.min(crop_1))
#     crop_2=(crop_2-np.min(crop_2))/(np.max(crop_2)-np.min(crop_2))
#     crop_3=(crop_3-np.min(crop_3))/(np.max(crop_3)-np.min(crop_3))
#     crop_4=(crop_4-np.min(crop_4))/(np.max(crop_4)-np.min(crop_4))
    
#     ssim12=ssim(crop_1,crop_2)
#     ssim13=ssim(crop_1,crop_3)
#     ssim14=ssim(crop_1,crop_4)
#     ssim23=ssim(crop_2,crop_3)
#     ssim24=ssim(crop_2,crop_4)
#     ssim34=ssim(crop_3,crop_4)
#     ssim_all=[ssim12,ssim13,ssim14,ssim23,ssim24,ssim34]
#     #print('ssim12: {} | ssim13: {} | ssim14: {} | ssim23: {} | ssim24: {} | ssim34: {}'.format(ssim12,ssim13,ssim14,ssim23,ssim24,ssim34))
#     ssim_min=np.min(ssim_all)
#     #print(ssim_min)
#     isnoise=True
#     if ssim_min<0.8:
#        return False
#     print('ssim12: {} | ssim13: {} | ssim14: {} | ssim23: {} | ssim24: {} | ssim34: {}'.format(ssim12,ssim13,ssim14,ssim23,ssim24,ssim34))
    #std_global=np.std(arr)
    std_global=310 #95
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
    #print('True :: std_global: {} | std_1: {} | std_2: {} | std_3: {} | std_4: {} '.format(std_global,std_1,std_2,std_3,std_4))
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
# for input_file in input_list:
#     print(input_file)
#     print(ii,'/',len(input_list))
#     ii=ii+1
#     #print(os.path.join(denoised_dir,input_file))
#     denoised_img=mrcfile.open(os.path.join(denoised_dir,input_file),permissive=True)
#     denoised_img=denoised_img.data
#     denoised_img=(denoised_img-np.min(denoised_img))/(np.max(denoised_img)-np.min(denoised_img))*255
#     denoised_img2=denoised_img.copy()

#     raw_img=mrcfile.open(os.path.join(raw_dir,input_file),permissive=True)
#     raw_img=raw_img.data
#     raw_img=(raw_img-np.min(raw_img))/(np.max(raw_img)-np.min(raw_img))*255
#     #img = img.astype(np.uint8)
    
#     w,h=denoised_img.shape
#     noise_patch_path=os.path.join(noise_dir,input_file)
#     draw_noise_path=os.path.join(draw_dir,input_file)
#     noise_patch_n=0
#     noise_list=[]
#     noise_max=[]
#     x_l=[]
#     y_l=[]
#     for x in range(0,w,step):
#         for y in range(0,h,step):
#             if x+box_size>w or y+box_size>h:
#                 continue
            
#             denoised_arr=denoised_img[x:x+box_size,y:y+box_size]
#             raw_arr=raw_img[x:x+box_size,y:y+box_size]
#             noise=is_noise(denoised_arr,x,x+box_size,y,y+box_size)
#             if noise[0]:
#                 #print(x,y)
#                 noise_list.append(raw_arr)
#                 noise_max.append(noise[1])
#                 x_l.append(x)
#                 y_l.append(y)
#                 cv2.rectangle(denoised_img, (y,x), (y+box_size,x+box_size), (0, 0, 255), 2)
#                 #save noise patch
#                 #with mrcfile.new(noise_patch_path[:-4]+'_'+str(noise_patch_n)+'.mrc',overwrite=True) as noise_patch:
#                      #noise_patch.set_data(arr)
#     print(len(noise_list))
#     print(len(noise_max))
#     print(len(x_l))
#     print(len(y_l))
#     #in_50_indices = np.argsort(noise_max)[:20]
#     num=int(len(noise_max)/5)
#     if len(noise_max)>150: 
#         #print('min_50_indices:',min_50_indices)
#         for i in np.argsort(noise_max)[:num]:
#             with mrcfile.new(noise_patch_path[:-4]+'_'+str(noise_patch_n)+'.mrc',overwrite=True) as noise_patch:
#                  noise_patch.set_data(noise_list[i])
#             #draw noise
#             cv2.rectangle(denoised_img2, (y_l[i],x_l[i]), (y_l[i]+box_size,x_l[i]+box_size), (0, 0, 255), 2)
#             noise_patch_n+=1
#         #save draw noise
#         with mrcfile.new(draw_noise_path,overwrite=True) as draw_noise:
#             draw_noise.set_data(denoised_img2)
#     else:
#         #print('min_50_indices:',min_50_indices)
#         for i in np.argsort(noise_max):
#             with mrcfile.new(noise_patch_path[:-4]+'_'+str(noise_patch_n)+'.mrc',overwrite=True) as noise_patch:
#                  noise_patch.set_data(noise_list[i])
#             #draw noise
#             cv2.rectangle(denoised_img2, (y_l[i],x_l[i]), (y_l[i]+box_size,x_l[i]+box_size), (0, 0, 255), 2)
#             noise_patch_n+=1
#         #save draw noise
#         with mrcfile.new(draw_noise_path,overwrite=True) as draw_noise:
#             draw_noise.set_data(denoised_img2)

def process_image(input_file):
    #img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    denoised_img=cv2.imread(os.path.join(denoised_dir,input_file), cv2.IMREAD_GRAYSCALE)
    #denoised_img=denoised_img.data
    denoised_img=(denoised_img-np.min(denoised_img))/(np.max(denoised_img)-np.min(denoised_img))*255
    denoised_img2=denoised_img.copy()

    raw_img=cv2.imread(os.path.join(raw_dir,input_file), cv2.IMREAD_GRAYSCALE)
    #raw_img=raw_img.data
    raw_img=(raw_img-np.min(raw_img))/(np.max(raw_img)-np.min(raw_img))*255
    #img = img.astype(np.uint8)
    
    w,h=denoised_img.shape
    noise_patch_path=os.path.join(noise_dir,input_file)
    draw_noise_path=os.path.join(draw_dir,input_file)
    noise_patch_n=0
    noise_list=[]
    noise_max=[]
    x_l=[]
    y_l=[]
    for x in range(0,w,step):
        for y in range(0,h,step):
            if x+box_size>w or y+box_size>h:
                continue
            
            denoised_arr=denoised_img[x:x+box_size,y:y+box_size]
            raw_arr=raw_img[x:x+box_size,y:y+box_size]
            noise=is_noise(denoised_arr,x,x+box_size,y,y+box_size)
            if noise[0]:
                #print(x,y)
                noise_list.append(raw_arr)
                noise_max.append(noise[1])
                x_l.append(x)
                y_l.append(y)
                cv2.rectangle(denoised_img, (y,x), (y+box_size,x+box_size), (255, 255, 255), 3)
                #save noise patch
                #with mrcfile.new(noise_patch_path[:-4]+'_'+str(noise_patch_n)+'.mrc',overwrite=True) as noise_patch:
                     #noise_patch.set_data(arr)
    print(len(noise_list))
    #print(len(noise_max))
    #print(len(x_l))
    #print(len(y_l))
    #in_50_indices = np.argsort(noise_max)[:20]
    num=int(len(noise_max)/4)
    #num=30
    if 250>len(noise_max)>100: 
        
        #print('min_50_indices:',min_50_indices)
        for i in np.argsort(noise_max)[:num]:
            cv2.imwrite(noise_patch_path[:-4]+'_'+str(i)+'.jpg', noise_list[i], [cv2.IMWRITE_JPEG_QUALITY, 75]);
            #with mrcfile.new(noise_patch_path[:-4]+'_'+str(i)+'.mrc',overwrite=True) as noise_patch:
                 #noise_patch.set_data(noise_list[i])
                
            
            cv2.rectangle(denoised_img2, (y_l[i],x_l[i]), (y_l[i]+box_size,x_l[i]+box_size), (0, 0, 255), 2)
            noise_patch_n+=1
        
        cv2.imwrite(draw_noise_path, denoised_img2, [cv2.IMWRITE_JPEG_QUALITY, 75]);
        #with mrcfile.new(draw_noise_path,overwrite=True) as draw_noise:
            #draw_noise.set_data(denoised_img2)
    elif len(noise_max)>=250:
        cv2.imwrite(draw_noise_path, denoised_img2, [cv2.IMWRITE_JPEG_QUALITY, 75]);
    else:
        #print('min_50_indices:',min_50_indices)
        for i in np.argsort(noise_max):
            cv2.imwrite(noise_patch_path[:-4]+'_'+str(i)+'.jpg', noise_list[i], [cv2.IMWRITE_JPEG_QUALITY, 75]);
            
            #with mrcfile.new(noise_patch_path[:-4]+'_'+str(i)+'.mrc',overwrite=True) as noise_patch:
                 #noise_patch.set_data(noise_list[i])
                    
            
            cv2.rectangle(denoised_img2, (y_l[i],x_l[i]), (y_l[i]+box_size,x_l[i]+box_size), (0, 0, 255), 2)
            noise_patch_n+=1
        #save draw noise
        cv2.imwrite(draw_noise_path, denoised_img2, [cv2.IMWRITE_JPEG_QUALITY, 75]);
        #with mrcfile.new(draw_noise_path,overwrite=True) as draw_noise:
            #draw_noise.set_data(denoised_img2)

pool = multiprocessing.Pool(processes=16)
#print(input_list)
pool.map(process_image, input_list[:])

# 关闭进程池
pool.close()
pool.join()