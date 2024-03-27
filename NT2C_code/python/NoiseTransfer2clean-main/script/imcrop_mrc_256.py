import numpy as np
import os
import mrcfile
import zipfile
#from  EMAN2 import *
import sys,os,getopt
#from matplotlib import pyplot as plt
import cv2
#from skimage import io
#from mpi4py  import MPI
import mrcfile
import zipfile
import numpy as np
import gzip

def norm(img):
    imax = np.max(img)
    imin = np.min(img)
    a = float(1) / (imax - imin)
    b = (-1) * a * imin

    sizex = img.shape[0]
    sizey = img.shape[1]
    arr = np.zeros([sizex, sizey])
    arr = a * img + b

    return arr

def getId(fname):
    lines = fname.split('/')
    fname = lines[-1]
    lines = fname.split('.')
    stop = -1 * (len(lines[-1]) + 1)
    fid = fname[:stop]
    print(fid)
    return fid
def cutImg(fin, img, size, step, dout, zipf):
    sizex, sizey = img.shape
    f0 = getId(fin)
    if sizex < size:
        size = sizex
    oldx = -1000
    oldy = -1000
    start = 0
    for x in range(start, sizex, step):
        for y in range(start, sizey, step):
            startx = x
            starty = y
            if startx + size > sizex:
                startx = sizex - size
            if starty + size > sizey:
                starty = sizey - size
            if oldx != startx or oldy != starty:
                arr = img[startx:startx+size, starty:starty+size]
                arr = norm(arr)
                fname = f0 + '.' + str(startx) + '.' + str(starty) + '.' + str(size) + '.mrc'
                # 创建一个新的MRC文件对象并设置数据
                mrc_out = mrcfile.new(fname, overwrite=True)
                mrc_out.set_data(arr)
                mrc_out.close()

                # 将MRC文件添加到Zip文件中
                zipf.write(fname)
                os.remove(fname)  # 删除临时的MRC文件

                oldx = startx
                oldy = starty

# 主程式
din = sys.argv[1]
dout_cut = sys.argv[2]
list = os.listdir(din)
step = 128
size = 256

output_zip = os.path.join(dout_cut, '256_clean.zip')
with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_LZMA) as zipf: #.ZIP_DEFLATED
    for i in range(0, len(list)):
        path = os.path.join(din, list[i])
        if os.path.isfile(path):
            with mrcfile.open(path, permissive=True) as em:
                img = em.data
                #img=img[0,:,:] #模擬用
                print('原圖大小:', img.shape)
                cutImg(path, img, size, step, dout_cut, zipf)
