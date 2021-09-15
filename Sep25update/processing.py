from pycocotools.coco import COCO
%matplotlib inline
import matplotlib.pyplot as plt
import cv2
from PIL import Image 
import numpy as np
import pickle
import os
import numpy as np
import random
from PIL import ImageTk
from IPython.display import display
import glob
import shutil

domain = 'val/'

# 定义变量
dataroot = 'FLIR_ADAS_1_3/'

jsonfile = dataroot + domain + 'thermal_annotations.json'
coco = COCO(jsonfile) # coco class to view jsonfile

import pickle

def view_FLIR(ind):
    imInfo = coco.imgs[ind]
    annIds = coco.getAnnIds(imgIds=imInfo['id'])
    imgfile = dataroot + domain + imInfo['file_name']

    print(f'{imInfo} \ncorresbonding annids is\n{annIds}\n')

    anns = coco.loadAnns(annIds)
    if anns:
        print(anns[0])

    img = cv2.imread(imgfile)

    for ann in anns:
        x, y, w, h = ann['bbox']
        cv2.rectangle(img, (x,y), (x + w, y + h), (255,0,0), 2)
        cat = coco.loadCats(ann['category_id'])[0]['name']
        cv2.putText(img, cat, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    plt.imshow(img)
    plt.show()

#convert coco bbox format to yolo bbox format (x_c_n,y_c_n,w_n,h_n), normalized center point coordinates; normalized weight/height 
def converter(size,bbox):
    #bbox = [x,y,w,h];(x,y) is the left top point of the bounding box
    x = (bbox[0]+bbox[2]*0.5)/size[0]
    y = (bbox[1]+bbox[3]*0.5)/size[1]
    w = bbox[2]/size[0]
    h = bbox[3]/size[1]
    return (x,y,w,h)

def copy_images(src,dst,ran,coco):
    for newidx, imageidx in enumerate(ran):
        img = coco.loadImgs(ids = [imageidx])
        file_name = img[0]['file_name']
        src_dir = src + file_name
        dst_dir = dst.format(newidx+1)
        shutil.copy(src_dir,dst_dir)

def save_labels(size,coco,Idx,domain):
    for imgidx in Idx:
        img = coco.loadImgs(ids = [imgidx])
        file_name = img[0]['file_name']
        name = file_name[-15:-5]
        f = open('FLIR_people_select/' + domain + 'labels/'+ name + '.txt','w')
        annid = coco.getAnnIds(imgIds=[imgidx])
        anns = coco.loadAnns(ids = annid)
        for ann in anns:
            bbox = ann['bbox']
            cat = ann["category_id"]-1
            (x,y,w,h) = converter(size,bbox)
            f.write('{:d} {:f} {:f} {:f} {:f}\n'.format(cat,x,y,w,h))
        f.close()



## save Val images

### Copy valid val images to the folder

ran = FLIR_people_select_valIdx 
src_val = "FLIR_ADAS_1_3/val/"
dst_val = "FLIR_people_select/val/images/"

for newidx, imageidx in enumerate(ran):
    img = coco.loadImgs(ids = [imageidx])
    file_name = img[0]['file_name']
    name = file_name[-15:]
    print(name)
    dst_dir = dst_val + name
    src_dir = src_val+file_name
    shutil.copy(src_dir,dst_dir)


FLIR_people_select_valIdx

with open ('FLIR_people_select_valIdx', 'rb') as fp:
     FLIR_people_select_valIdx = pickle.load(fp)

### Copy valid training images to the folder

'''
ran = FLIR_people_select_trainIdx
src_train = "FLIR_ADAS_1_3/train/"
dst_train = "FLIR_people_select/train/images/FLIR_{:05d}.jpeg"
copy_images(src_train,dst_train,ran)
'''





## save train images

ran = FLIR_people_select_trainIdx 
src_train = "FLIR_ADAS_1_3/train/"
dst_train = "FLIR_people_select/train/images/"

for newidx, imageidx in enumerate(ran):
    img = coco.loadImgs(ids = [imageidx])
    file_name = img[0]['file_name']
    name = file_name[-15:]
    print(name)
    dst_dir = dst_train + name
    src_dir = src_train + file_name 
    shutil.copy(src_dir,dst_dir)

with open ('FLIR_people_select_trainIdx', 'rb') as fp:
     FLIR_people_select_trainIdx = pickle.load(fp)



#with open ('FLIR_people_select_trainIdx', 'rb') as fp:
#     FLIR_people_select_trainIdx = pickle.load(fp)
with open ('FLIR_people_select_valIdx', 'rb') as fp:
     FLIR_people_select_valIdx = pickle.load(fp)

len(FLIR_people_select_valIdx)

len(FLIR_people_select_trainIdx)

anns = coco.loadAnns(ids = [FLIR_people_select_trainIdx[3]])
bbox = anns[0]['bbox']

### save txt files

# train


import pickle
with open ('FLIR_people_select_trainIdx', 'rb') as fp:
     FLIR_people_select_trainIdx = pickle.load(fp)


with open ('FLIR_people_select_valIdx', 'rb') as fp:
     FLIR_people_select_valIdx = pickle.load(fp)

size = [640,512]
if domain == 'train/':
    save_labels(size,coco,FLIR_people_select_trainIdx,domain)

# val
size = [640,512]
if domain == 'val/':
    save_labels(size,coco,FLIR_people_select_valIdx,domain)
