from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import transforms
import cv2
from pycocotools.coco import COCO

import cv2
from PIL import Image 
import random
import os
import numpy as np
import random
import skimage.transform

def adjust_contrast_bri(contrast,brightness,img_ori):
    img = img_ori
    img = np.int16(img)
    img = img * (contrast/127+1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    return img

# rndomly rorate the image within certain angle
def rand_rotate(patch,angle_ran):
    angle1 = np.random.randint(-angle_ran,angle_ran)
    rotated_patch = skimage.transform.rotate(patch,angle = angle1)
    #rotated_image1 = 
    return rotated_patch


def yolo_cc(file): #from yolo txt file to coco format
    x = (x_yl-w_yl/2)*width #weight is the size of the image
    y = (y_yl*height-h_yl/2)*height
    w = w_yl*width
    h = h_yl*height
    return x,y,w,h

def view_FLIR(ind): #input an image id
    imInfo = coco.imgs[ind]
    annIds = coco.getAnnIds(imgIds=imInfo['id']) #get annotations id of this image
    imgfile = dataroot + domain + imInfo['file_name']

    print(f'{imInfo} \ncorresbonding annids is\n{annIds}\n')

    anns = coco.loadAnns(annIds)#load the annotations information according to the anns id
    if anns:
        print(anns[0]) #show the first annotations
    
    img = cv2.imread(imgfile) #open the image file
    im_ori = cv2.imread(imgfile)

    for ann in anns:
        x, y, w, h = ann['bbox']
        cv2.rectangle(img, (x,y), (x + w, y + h), (255,0,0), 2)
        cat = coco.loadCats(ann['category_id'])[0]['name']
        cv2.putText(img, cat, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    plt.imshow(img)
    plt.show()
    return im_ori

def multiple_gussians(centers, s, sig, box_size, x, y):
    patch_sum = np.zeros((box_size[1],box_size[0]))
    #patch_sum = np.repeat(zeros[...,np.newaxis],3,axis = 2)
    for idx in range(0,len(centers),2):
        xc = (1-centers[idx])*x
        yc = (1-centers[idx+1])*y
        print(xc,yc)
        patch = gaussian(xc,yc, s, sig, box_size)
        patch_sum = patch_sum + patch
    return patch_sum

#uppder body patch, 1/5
def patch_range(bbox,sigma):
    #gaussian distribution beome zero after 2*sd
    sigma_d = 2*sigma
    #wb,hb are the highet width of the bonding box
    xbb,ybb,wbb,hbb = bbox
    #the patch box
    sigma = int(sigma)
    if sigma and (int(np.floor(ybb+hbb/7))-int(sigma_d))>0 :
            x,y,w,h = int(xbb-sigma_d) if (xbb-sigma_d)>0 else xbb ,int(np.floor(ybb+hbb/7) - sigma_d) if int(np.floor(ybb+hbb/7)-sigma_d)>0 else ybb ,int(wbb+2*sigma_d) if int(xbb-sigma_d+wbb+2*sigma_d) <= 640 and int(xbb-sigma_d+wbb+2*sigma_d)> 0 else wbb,int(np.floor(hbb/5)+2*sigma_d) if int(np.floor(ybb+hbb/7)- sigma_d + np.floor(hbb/5)+2*sigma_d) <= 512 and int(np.floor(ybb+hbb/7)-sigma_d + np.floor(hbb/5)+2*sigma_d)>0 else hbb
    elif sigma==0:
        x,y,w,h = int(xbb),int(np.floor(ybb+hbb/7)),int(wbb),int(np.floor(hbb/5))
    return x,y,w,h

#txt file: label, center_x, certer_y,w,h
def gaussian(xc,yc, s, sig, box_size): #xc, yc are coordinates respective to the bbox, xc<w;yc<h
    patch_w = box_size[1]
    patch_h = box_size[0]
    gus_patch = np.zeros((patch_w,patch_h))
    for xb in range(patch_w):
        for yb in range(patch_h):            
            r = (xb-xc)**2+(yb-yc)**2
            gus_patch[xb][yb] = s*np.exp(-r/(2*sig**2))
    return gus_patch

def patched_image(im,patch,x,y,w,h):
    im_p = im.astype('float')
    print(im_p[y:y+h,x:x+w].shape,patch.shape)
    #border test
    p = [a + b for a, b in zip(im_p[y:y+h,x:x+w], patch)]
    for each,i in enumerate(p) :
        if each > 255:
            p[i] = 255
    im_p[y:y+h,x:x+w] = p
    #im = np.array(im)
    #im_dp = im.astype(float)
    #im_dp[y:y+h,x:x+w,:] = im_dp[y:y+h,x:x+w,:] + patch
    #im_dp = [a + b for a, b in zip(im[y:y+h,x:x+w,:], patch)]
    return im_p
