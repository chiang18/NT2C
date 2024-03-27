#from skimage.measure import compare_ssim as ssim
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
from matplotlib import pyplot as plt
import mrcfile
import mahotas
import sys,os
import matplotlib.pyplot as plt
import pywt
import zipfile


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
    '''
    crop_1=(crop_1-np.min(crop_1))/(np.max(crop_1)-np.min(crop_1))
    crop_2=(crop_2-np.min(crop_2))/(np.max(crop_2)-np.min(crop_2))
    crop_3=(crop_3-np.min(crop_3))/(np.max(crop_3)-np.min(crop_3))
    crop_4=(crop_4-np.min(crop_4))/(np.max(crop_4)-np.min(crop_4))
    
    ssim12 = ssim(crop_1, crop_2, data_range=crop_1.max() - crop_1.min())
    ssim13 = ssim(crop_1, crop_3, data_range=crop_1.max() - crop_1.min())
    ssim14 = ssim(crop_1, crop_4, data_range=crop_1.max() - crop_1.min())
    ssim23 = ssim(crop_2, crop_3, data_range=crop_2.max() - crop_2.min())
    ssim24 = ssim(crop_2, crop_4, data_range=crop_2.max() - crop_2.min())
    ssim34 = ssim(crop_3, crop_4, data_range=crop_3.max() - crop_3.min())
    
    ssim12=ssim(crop_1,crop_2)
    ssim13=ssim(crop_1,crop_3)
    ssim14=ssim(crop_1,crop_4)
    ssim23=ssim(crop_2,crop_3)
    ssim24=ssim(crop_2,crop_4)
    ssim34=ssim(crop_3,crop_4)
    
    isnoise=True
    ssim_all=[ssim12,ssim13,ssim14,ssim23,ssim24,ssim34]
    #print('ssim12: {} | ssim13: {} | ssim14: {} | ssim23: {} | ssim24: {} | ssim34: {}'.format(ssim12,ssim13,ssim14,ssim23,ssim24,ssim34))
    ssim_min=np.min(ssim_all)
    print('min',ssim_min)
    if ssim_min<0.1: #0.8 0.109
        isnoise=False
        return isnoise
    return isnoise
    '''
    #std_global=np.std(arr)*10.2
    std_global=135 #370 topaz:175
    #print(std_global)
    std_1=np.std(crop_1)
    std_2=np.std(crop_2)
    std_3=np.std(crop_3)
    std_4=np.std(crop_4)
    #print('std_global: {} | std_1: {} | std_2: {} | std_3: {} | std_4: {} '.format(std_global,std_1,std_2,std_3,std_4))

    isnoise=True
    if std_1>0.1*std_global:
        isnoise=False
        return isnoise
    if std_2>0.1*std_global:
        isnoise=False
        return isnoise
    if std_3>0.1*std_global:
        isnoise=False
        return isnoise
    if std_4>0.1*std_global:
        isnoise=False
        return isnoise
    print('True :: std_global: {} | std_1: {} | std_2: {} | std_3: {} | std_4: {} '.format(std_global,std_1,std_2,std_3,std_4))
    return isnoise
    
denoised_dir=sys.argv[1]  #micrographs
raw_dir=sys.argv[2]  #raw micrographs
noise_dir=sys.argv[3]  #noise patches
draw_dir=sys.argv[4] #draw noise 


box_size=256
par_size=200
step=int(0.02*box_size)
#print('step:  {}'.format(step))
noise_patch_num=0

input_list=sorted(os.listdir(denoised_dir))[22:23]
print(input_list)
# output_zip = os.path.join(noise_dir, 'noise_patches2.zip')
# draw_noise_zip = os.path.join(draw_dir, 'draw_noise2.zip')
# with zipfile.ZipFile(draw_noise_zip, 'w', zipfile.ZIP_DEFLATED) as draw_zipf, zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
i=1
for input_file in input_list:
  # print(os.path.join(denoised_dir,input_file))
  print(input_file)
  print(i,'/',len(input_list))
  i=i+1
  # print(input_file[:-4])
  denoised_img=mrcfile.open(os.path.join(denoised_dir,input_file),permissive=True)
  denoised_img=denoised_img.data
  denoised_img=(denoised_img-np.min(denoised_img))/(np.max(denoised_img)-np.min(denoised_img))*255

  raw_img=mrcfile.open(os.path.join(raw_dir,input_file),permissive=True)
  raw_img=raw_img.data
  raw_img=(raw_img-np.min(raw_img))/(np.max(raw_img)-np.min(raw_img))*255
  #img = img.astype(np.uint8)

  w,h=denoised_img.shape
  noise_patch_path=os.path.join(noise_dir,input_file)
  draw_noise_path=os.path.join(draw_dir,input_file)
  noise_patch_n=0
  for x in range(0,w,step):
      for y in range(0,h,step):
          if x+box_size>w or y+box_size>h:
              continue
          #print(x,y)
          denoised_arr=denoised_img[x:x+box_size,y:y+box_size]
          #print('raw_img_shape',raw_img.shape)
          #raw_arr=raw_img[0,x:x+box_size , y:y+box_size] 模擬
          raw_arr=raw_img[x:x+box_size , y:y+box_size] #真實
          noise=is_noise(denoised_arr,x,x+box_size,y,y+box_size)
          if noise:
              #save noise patch
              #with mrcfile.new(noise_patch_path[:-4]+'_'+str(noise_patch_n)+'.mrc',overwrite=True) as noise_patch:
                  #noise_patch.set_data(arr)
              #print('shape:',raw_img.shape) #1,1024,1024
              #print('shape:',y)
              #print('shape:',raw_arr)
              #print('shape:',raw_arr.shape)
              
              #with mrcfile.new(noise_patch_path[:-4]+'_'+str(noise_patch_n)+'.mrc',overwrite=True) as noise_patch:
                  #noise_patch.set_data(raw_arr)
                    
              #print('input_file',noise_patch_path[:-4])
            
#               with  mrcfile.new(input_file[:-4]+'_'+str(noise_patch_n)+'.mrc', overwrite=True) as mrc_out:
#                   mrc_out.set_data(raw_arr.astype(np.float32))
#                   mrc_out.close()
#                   zipf.write(input_file[:-4]+'_'+str(noise_patch_n)+'.mrc')
#                   os.remove(input_file[:-4]+'_'+str(noise_patch_n)+'.mrc')

              #draw noise
              cv2.rectangle(denoised_img, (y,x), (y+box_size,x+box_size), (0, 0, 255), 2)
              noise_patch_n+=1
            
  # save draw noise
  with mrcfile.new(draw_noise_path,overwrite=True) as draw_noise:
      draw_noise.set_data(denoised_img)

#   with  mrcfile.new(input_file, overwrite=True) as draw_out:
#       draw_out.set_data(denoised_img)
#       draw_out.close()
#       draw_zipf.write(input_file)
#       os.remove(input_file)




