import numpy as np
from pathlib import Path
import os
import cv2
import pickle
from pycocotools.coco import COCO
import patch as pat
import shutil

import skimage.transform
import numpy as np
from PIL import Image

#https://blog.csdn.net/qq_38048756/article/details/103208834
domain = 'train/'

dataroot = 'FLIR_people_select/' 

jsonroot = 'FLIR_ADAS_1_3/' 
jsonfile = jsonroot + domain + 'thermal_annotations.json'
coco = COCO(jsonfile)

# path to store file
# path =

#domain = 'train/'


!git clone https://github.com/ultralytics/yolov3  # clone repo
%cd yolov3
%pip install -qr requirements.txt  # install dependencies

import torch
from IPython.display import Image, clear_output  # to display images

clear_output()
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

### generate patches for a single image 
def valid_ann(ann):
    if ann['category_id'] == 1 and ann['bbox'][3]>120:
        return True
    else:
        return False

def generate_patch(idn,path_in_str,s,sig,angle,centers):
    imInfo = coco.imgs[idn]
    annIds = coco.getAnnIds(imgIds=imInfo['id']) #get annotations id of this image
    anns = coco.loadAnns(annIds)
    img = cv2.imread(path_in_str,0)
    for person_ann in anns:
        print(person_ann )
        if valid_ann(person_ann) == True:
            x,y,w,h = pat.patch_range(person_ann['bbox'],sig)
            print(x,y,w,h)
            #delete the rectangle when necessary
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            box_size = (w,h)
            patch = pat.multiple_gussians(centers,s,sig ,box_size, x ,y)
            print(patch.shape)
            
            name = path_in_str[-15:-5]
            save_path = 'patched_image/train/'+ name
            
            #patche one
            im_p = pat.patched_image(img,patch,x,y,w,h)
            im = Image.fromarray(np.uint8(im_p))
            im.save(save_path + '.jpeg','JPEG')
            
            #rotated patch
            patch_rotate = pat.rand_rotate(patch,angle)
            im_p = pat.patched_image(img, patch_rotate,x,y,w,h)
            im = Image.fromarray(np.uint8(im_p))
            im.save(save_path + '_r.jpeg','JPEG')
            
            #bright-changed patch
            patch_bright = pat.adjust_contrast_bri(contrast,brightness,patch)
            im_p = pat.patched_image(img, patch_bright ,x,y,w,h)
            im = Image.fromarray(np.uint8(im_p))
            im.save(save_path + '_b.jpeg','JPEG')
        else:
            continue

        
#generate patches for image in a domain
def generate_patches(centers,domain,s,sig,rotate_ran):
    if domain == 'train/':
        with open ('FLIR_people_select_trainIdx', 'rb') as fp:
            idxs = pickle.load(fp)
    elif domain =='val/':
        with open ('FLIR_people_select_valIdx', 'rb') as fp:
            idxs = pickle.load(fp)
    #patch_sum = patch.multiple_gussians(centers)
    path = dataroot + domain 
    pathlist = Path(path).rglob('*.jpeg')
    for path,idn in zip(pathlist,idxs):
        path_in_str = str(path)
        print(path_in_str)
        generate_patch(idn,path_in_str,s,sig,rotate_ran,centers)
        #save image in FLIR/patched_image/train
    
    #file path of the patched images  
    return 

def yolo_scores(path):
    pathlist = Path(path).rglob('*.txt')
    scores_yolo = list()
    images = list()
    count = 0
    for path in pathlist:
        count += 1
        image = list()
        f = open(path,'r')
        lines = f.readlines()
        for line in lines:
            num_list = line.split()
            #print(num_list)
            if num_list[0] == "0":
                image.append(float(num_list[-1]))  
        if count == 4:
            scores_yolo.append(images)
            images = list(')
            count = 0
        else:
            images.append(image)
    return scores_yolo
                
def yolov3_scores(centers):
    patch_images = generate_patches(centers) #path to folders store the images
    ...
    #use terminal command instead of function
    return scores # a list
def faster_rcnn_scores():
    generate_patches(centers)
    ...
    return scores # alist
def loss_function(centers):
    detector1_scores = yolov3_scores(centers)
    #detector2_scores = faster_rcnn_scores(centers)
    #L1_detector = 


# save patched image label for yolov3
if domain == 'train/':
    with open ('FLIR_people_select_trainIdx', 'rb') as fp:
        idxs = pickle.load(fp)
elif domain =='val/':
    with open ('FLIR_people_select_valIdx', 'rb') as fp:
        idxs = pickle.load(fp)
ran = idxs 
src_train = "FLIR_people_select/train/labels/"
dst_train = "patched_image/train/labels/"

for newidx, imageidx in enumerate(ran):
    img = coco.loadImgs(ids = [imageidx])
    file_name = img[0]['file_name']
    name = file_name[-15:-5] 
    print(name)
    src_dir = src_train + name + '.txt'
    dst_dir1 = dst_train + name + '.txt'
    shutil.copy(src_dir,dst_dir1)
    dst_dir2 = dst_train + name + '_b.txt'
    shutil.copy(src_dir,dst_dir2)
    dst_dir3 = dst_train + name + '_r.txt'
    shutil.copy(src_dir,dst_dir3)
    print(dst_dir1,dst_dir2,dst_dir3)

idxs

names = list()
'''
def generate_patch(path):
    pathlist = Path(path).rglob('*.jpeg')
    for path in pathlist:
        path_in_str = str(path)
        print(path)
        names.append(path_in_str)

dataroot = '/home/hwjin/FLIR/yolov3/FLIR_people_select/' + os.sep
generate_patch(dataroot + 'val/')

'''

centers = [0.05,0.3,0.35,0.6,0.67,0.9]

s = 10
sig = 5.07
rotate_ran = 20

contrast = 30
brightness = 50

generate_patches(centers,domain,s,sig,rotate_ran)


if domain == 'train/':
    with open ('FLIR_people_select_trainIdx', 'rb') as fp:
        idxs = pickle.load(fp)
        
test = list()
for id in idxs:
    imginfo = coco.imgs[id]
    test.append(imginfo['file_name'])

test

path = dataroot + domain 
pathlist = Path(path).rglob('*.jpeg')
for path,idn in zip(pathlist,idxs):
    print(path)

#yolov3 detect inference patched image
!python detect.py --weights yolov3.pt --img 640 --conf 0.25 --save-conf --save-txt --source patched_image/train/images

#Image(filename='runs/detect/exp/zidane.jpg', width=600)

path = 'runs/detect/exp11/labels'

    pathlist = Path(path).rglob('*.txt')
    scores_yolo = list()
    for path in pathlist:
        f = open(path,'r')
        lines = f.readlines()
        for line in lines:
            num_list = line.split()
            print(num_list)
            if num_list[0] == "0":
                scores_yolo.append(float(num_list[-1]))       

scores = yolo_scores(path)

scores

