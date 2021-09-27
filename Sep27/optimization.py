#!/usr/bin/env python
# coding: utf-8

# In[2]:


cd FLIR


# In[3]:


import numpy as np
from pathlib import Path
import os
import cv2
import pickle
from pycocotools.coco import COCO
import patch as pat
import shutil

import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
from PIL import Image

import random
import torch.nn as nn
import keras.backend as K
from torch import tensor 
import torch

import warnings
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

#https://blog.csdn.net/qq_38048756/article/details/103208834
domain = 'train/'

dataroot = 'FLIR_people_select/' 

jsonroot = 'FLIR_ADAS_1_3/' 
jsonfile = jsonroot + domain + 'thermal_annotations.json'
coco = COCO(jsonfile)

# path to store file
# path =

#domain = 'train/'


# In[4]:


import random


# In[5]:


import pickle
with open ('FLIR_people_select_trainIdx', 'rb') as fp:
     idxs = pickle.load(fp)


# In[6]:


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
            patch = multiple_gussians(centers,s,sig ,box_size, x ,y)
            print(patch.shape)
            
            name = path_in_str[-15:-5]
            save_path = 'patched_image/train/'+ name
            
            #patche one
            im_p = patched_image(img,patch,x,y,w,h)
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

def multiple_gussians(centers, s, sig, box_size, x, y):
    patch_sum = np.zeros((box_size[1],box_size[0]))
    #patch_sum = np.repeat(zeros[...,np.newaxis],3,axis = 2)
    for idx in range(0,len(centers),2):
        xc = x+ centers[idx]*box_size[0]
        yc = y+ centers[idx+1]*box_size[1]
        #print(xc,yc)
        
        #print(xc,yc, s, sig, box_size)
        
        patch = gaussian(xc,yc, s, sig, box_size)
        #print(patch)
        patch_sum = patch_sum + patch
    #print('patched_sum',patched_sum)
    return patch_sum

def gaussian(xc,yc, s, sig, box_size): #xc, yc are coordinat
    patch_w = box_size[1]
    patch_h = box_size[0]
    gus_patch = np.zeros((patch_w,patch_h))
    for xb in range(patch_w):
        for yb in range(patch_h):
            #print('gussian',xb,yb)
            r = np.sqrt((xb-xc)**2+(yb-yc)**2)
            #print('r',r)
            gus_patch[xb][yb] = s*np.exp(-r/(2*sig**2))
    #print(gus_patch)
    return gus_patch

#directly modify image, not store it
def model_patches(img,ind,centers,s,sig,rotate_ran):
    import tensorflow as tf
    from tensorflow.python.framework.ops import enable_eager_execution
    enable_eager_execution()
    imInfo = coco.imgs[ind]
    annIds = coco.getAnnIds(imgIds=imInfo['id']) #get annotations id of this image
    anns = coco.loadAnns(annIds)
    #img = cv2.imread(path_in_str,0)
    for person_ann in anns:
        if valid_ann(person_ann) == True:
            x,y,w,h = pat.patch_range(person_ann['bbox'],sig)
            
            #print(x,y,w,h)
            
            #delete the rectangle when necessary
            #cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            box_size = (w,h)
            
            #print(centers,s,sig ,box_size, x ,y)
            
            patch = multiple_gussians(centers,s,sig ,box_size, x ,y)
            #print(patch)
            #print(patch.shape)
            
            #patche one
            im_p = patched_image(img,patch,x,y,w,h)
            #im = tf.convert_to_tensor(im_p, dtype=None, dtype_hint=None, name=None)
            
            #im = Image.fromarray(np.uint8(im_p))
            #im.save(save_path + '.jpeg','JPEG')
            
            #rotated patch
            patch_rotate = pat.rand_rotate(patch,rotate_ran)
            #im_r = patched_image(img, patch_rotate,x,y,w,h)
            
            #im = Image.fromarray(np.uint8(im_p))
            #im.save(save_path + '_r.jpeg','JPEG')
            
            #bright-changed patch
            contrast = 10
            brightness = 5
            
            patch_bright = pat.adjust_contrast_bri(contrast,brightness,patch)
            #im_b = patched_image(img, patch_bright ,x,y,w,h)
            
            #im = Image.fromarray(np.uint8(im_p))
            #im.save(save_path + '_b.jpeg','JPEG')
            
            return im_p,patch_rotate,patch_bright
        else:
            continue

def patched_image(im,patch,x,y,w,h):
    #print(im.shape)
    patch = np.expand_dims(patch,axis = 0)
    patch = np.repeat(patch,3,axis = 0)
    im_p = im.astype('float')
    #print(im_p[:,y:y+h,x:x+w].shape,patch.shape)
    #border test
    p = [a + b for a, b in zip(im_p[:,y:y+h,x:x+w], patch)]
    for each,i in enumerate(p) :
        if each > 255:
            p[i] = 255

    im_p[:,y:y+h,x:x+w] = p
    #im = np.array(im)
    #im_dp = im.astype(float)
    #im_dp[y:y+h,x:x+w,:] = im_dp[y:y+h,x:x+w,:] + patch
    #im_dp = [a + b for a, b in zip(im[y:y+h,x:x+w,:], patch)]
    return im_p
    
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
            images = list()
            count = 0
        else:
            images.append(image)
    return scores_yolo
                
def yolov3_(centers):
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
    #detector2_scores = faster_rcnn_scores(centers)x
    #L1_detector =  


# In[7]:


#imgname = 'FLIR_00320.jpeg'
#img = cv2.imread(imgname)
#plt.imshow(img)


# In[8]:


def yolov3_score(img):
    import cv2
    from pytorchyolo import detect, models
    
    #img = cv2.imread('FLIR_08486.jpeg')
    #img convert type
    # Load the YOLO model
    model = models.load_model(
      "config/yolov3.cfg", 
      "weights/yolov3.weights")

    # Load the image as a numpy array

    # Convert OpenCV bgr to rgb
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Runs the YOLO model on the image 
    boxes = detect.detect_image(model, img)
    for each in boxes:
        if each[-1] == 0:
            score = each[-2]
    print(score)
# Output will be a numpy array in the following format:
# [[x1, y1, x2, y2, confidence, class]]

def input_preprocess():
    im_nd = image.numpy()
    im_nd = np.repeat(im_nd, 3, axis=2)
    yolov3_score(im_nd)


# ### Yolov3

# In[9]:


pwd


# In[10]:


cd yolov3


# In[11]:


#import library
from yolov3.utils import loss
from models import yolo
import models
from models.experimental import attempt_load
from torch import tensor 
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds,     fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size,     check_requirements, print_mutation, set_logging, one_cycle, colorstr

#import hyperparameter-hyp file
import yaml
with open('data/hyp.finetune.yaml') as f:
    hyp = yaml.safe_load(f)


# In[12]:


#import yolov3 model
model = yolo.Model('models/yolov3.yaml', ch=3, nc=80,anchors=hyp.get('anchors')).to('cuda')


# In[13]:


#create compute_loss class
from utils.loss import ComputeLoss
model.hyp = hyp
model.gr = 1
compute_loss = ComputeLoss(model)


# #### centers as parameters

# In[14]:


class opt2():
    def __init__(self):
        self.cache_images =1
        self.rect = True
        self.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        self.quad=False
        self.workers = 8
        self.image_weights = False
        self.single_cls = False


# In[15]:


opt=opt2()


# In[16]:


from utils.datasets import create_dataloader
import torch.optim as optim

train_path = 'FLIR_people_select/train/images'
imgsz = 640
batch_size = 16
gs = 8 #grid size


dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                        hyp=hyp, augment=False, cache=opt.cache_images, rect=opt.rect, rank=-1,
                                        world_size=opt.world_size, workers=opt.workers,
                                        image_weights=opt.image_weights, quad=opt.quad) #prefix=colorstr('train: '))
pbar = enumerate(dataloader)


# In[17]:


#centers as parameters
import torch.nn as nn
import numpy as np
import tensorflow as tf
import torch

pg0 = torch.nn.Parameter(data=torch.Tensor([0.3,0.4,0.15,0.66,0.31,0.89]), requires_grad=True)#mizer parameter groups
optimizer = optim.SGD([pg0], lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
#logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))


# In[18]:


params_group = optimizer.param_groups
centers = params_group[0]['params'][0]


# Sample, show images from dataloader

# In[19]:


#centers as parameters
from torch.cuda import amp
import torch
import torch.optim as optim
import matplotlib as mpl
mpl.use('Agg')

count = 0
optimizer.zero_grad()
for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
    imgs = imgs.to(torch.float)
    
    ind = idxs[count]
    
    #print(ind)
    s = 10
    sig = 5.07
    rotate_ran = 20
    
    params_group = optimizer.param_groups
    centers = params_group[0]['params'][0]
    
    imgs_processed = list()

    print(imgs.shape)
    for i in range(16):
        #img = torch.movedim(imgs[i], 0, 2)
        im_np = imgs[i].detach().numpy()
       # print(im_np.shape)
        centers_np = centers.detach().numpy()
        im,im_r,im_b = model_patches(im_np,ind,centers_np,s,sig,rotate_ran)
        imgs_processed.append(im)
        #img = (im/255.).astype(float)
        #img = img.swapaxes(0,1)
        #img = img.swapaxes(1,2)
        #plt.imshow(img)
        
    
    count+=1

    imgs_processed = torch.tensor(imgs_processed).float().to('cuda')
    
    pred = model(imgs_processed)  # forward
    for i,p in enumerate(pred):    
        print(pred[i].shape)
        
    #loss = x + 0 * self.__hidden__(x)
    loss,lobj, loss_items = compute_loss(pred, targets.to('cuda'))
    if opt.quad:
        loss *= 4.
        loss_items *= 4.
    print('loss:',loss)
    print('obj_loss:',loss_items[1])
    scaler = amp.GradScaler()
    scaler.scale(lobj).backward()
    #if ni % accumulate == 0:
    scaler.step(optimizer)  # optimizer.step
    scaler.update()
    optimizer.zero_grad()


# In[85]:


import gc

gc.collect()

torch.cuda.empty_cache()


# In[21]:


lobj


# In[78]:


loss


# In[72]:


img


# In[33]:


import matplotlib as mpl
import torch
mpl.use('Agg')

pbar = enumerate(dataloader)

get_ipython().run_line_magic('matplotlib', 'inline')
from  matplotlib import pyplot as plt 
for i, (imgs, targets, paths, _) in pbar:
    print(imgs.shape)
    imgs = imgs.to(torch.float)
    
    for i in range(10):
        img = (imgs[i]/255)
        img = torch.movedim(img, 0, -1)
        plt.imshow(img)
        plt.show()
        break
    #arr = np.ndarray(img)#This is your tensor
    #arr_ = np.squeeze(arr) # you can give axis attribute if you wanna squeeze in specific dimension
    #plt.imshow(arr_)
    


# In[19]:


#im1_plt = im_np
#print(im1_plt)
#img1 = np.moveaxis(im1_plt,0,-1)
#img1 = torch.movedim(img, 0, -1)
#im2_plt = im
#print(im2_plt)
#img2 = np.moveaxis(im2_plt,0,-1) # patched image, what we want
#plt.imshow(img2.astype('uint8'))


# #### Output format of yolov3
# 
# 16 means batch size, 3 means three output layers with different feature map size; 64x80 = 16*(4x5) denoting number of anchors[bounding box], the same scale for image size 512x640. The rest 85 denote class.

# In[104]:


pred[1][1][1][1][10]
            


# In[80]:


for i,p in enumerate(pred):    
    print(pred[i].shape)


# #### Update weights and biases

# In[12]:


pwd


# In[13]:


class opt2():
    def __init__(self):
        self.cache_images =1
        self.rect = 0
        self.world_size =1
        self.quad=1
        self.workers = 0
        self.image_weights = 0
        self.single_cls = 0


# In[14]:


opt=opt2()


# In[15]:


from utils.datasets import create_dataloader
import torch.optim as optim

train_path = 'FLIR_people_select'
imgsz = 512
batch_size = 1
gs = 1


dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                        hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=-1,
                                        world_size=opt.world_size, workers=opt.workers,
                                        image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
pbar = enumerate(dataloader)


# In[25]:


#optimizer
import torch.nn as nn

pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
for k, v in model.named_modules():
    if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
        pg2.append(v.bias)  # biases
    if isinstance(v, nn.BatchNorm2d):
        pg0.append(v.weight)  # no decay
    elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
        pg1.append(v.weight)  # apply decay
optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
optimizer.add_param_group({'params': pg2})  # add pg2 (biases)


# In[16]:


for k, v in model.named_modules():
    print('k',k)
    print('v',v)


# In[21]:


import torch.nn as nn

for k, v in model.named_modules():
    if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
        print(v.bias.shape)  # biases
    if isinstance(v, nn.BatchNorm2d):
        print(v.weight.shape)  # no decay
    elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
        print(v.weight.shape)  # apply decay


# In[5]:


# Download COCO128
import torch
from IPython.display import Image, clear_output  # to display images

#clear_output()
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
#torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip', 'tmp.zip')


# In[62]:


#normal
from torch.cuda import amp
import torch
import torch.optim as optim


optimizer.zero_grad()
for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
    #print('!!!!!!!!!!!!!!!!!!!!!pbar:',pbar)
    #rint('!!!!!!!!!!!!!!!!!!!!!imgs:',imgs)
    #print('!!!!!!!!!!!!!!!!!!!!!targets:',targets)
    #print('!!!!!!!!!!!!!!!!!!!!!paths:',paths)
    #print('!!!!!!!!!!!!!!!!!!!!!_:',_)
    #import torchvision
    #images = torchvision.transforms.functional.to_tensor(imgs)
    imgs = imgs.to(torch.float)
    print('image shape:',imgs.shape)
    pred = model(imgs)  # forward
    ##print('pred:',pred)
    loss, loss_items = compute_loss(pred, targets.to('cuda'))
    if opt.quad:
        loss *= 4.
    print('loss:',loss)
    print('obj_loss:',loss_items[1])
    scaler = amp.GradScaler()
    scaler.scale(loss).backward()
    #if ni % accumulate == 0:
    scaler.step(optimizer)  # optimizer.step
    scaler.update()
    optimizer.zero_grad()


# In[ ]:





# In[13]:


get_ipython().system('git https://github.com/Liu-Yicheng/YOLOv3.git')


# In[ ]:


#from torch import tensor
#imgp = np.expand_dims(img,axis = 0)
#imagep = tensor(imgp)
#pred = model(imagep)[0]


# In[89]:


class opt():
    def __init__(self,source,weights):
        self.source = source
        self.weights = weights
        self.view_img = True
        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det = 1000
        self.project = 'runs/detect'
        self.device = 'cuda'
        self.save_txt = 1
        self.nosave=0
        self.save_conf = 1
        self.name = 'exp'
        self.exist_ok=0
        self.augment = 0
        self.classes = 0
        self.hide_conf = False
        self.hide_labels = False
        self.line_thickness = 3
        self.update = 0
        self.agnostic_nms = 0
        self.save_crop = 0


# In[90]:


from yolov3.detect import detect
import argparse

a = opt(source = 'FLIR_00320.jpeg',weights = 'yolov3.pt') 
box = detect(a)


# ### PyTorch-YOLOv3

# In[1]:


cd PyTorch-YOLOv3


# In[2]:


from pytorchyolo import detect, models
from pytorchyolo.utils.loss import compute_loss
from pytorchyolo.models import load_model


# In[90]:


model = models.load_model(
  "config/yolov3.cfg", 
  "weights/yolov3.weights")


# In[80]:


model


# In[36]:


# Compute loss
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(model_fn(x), y)
# If attack is targeted, minimize loss of target label rather than maximize loss of correct label
if targeted:
    loss = -loss

# Define gradient of loss wrt input
loss.backward()
optimal_perturbation = optimize_linear(x.grad, eps, norm)


# In[23]:


imgname = 'FLIR_00320.jpeg'
img = cv2.imread(imgname)


# In[ ]:



with tf.GradientTape() as g:
    g.watch(x)
    with tf.GradientTape() as gg:
        gg.watch(x)
        y = x * x### Gradient tape to get derivative of loss function
dy_dx = gg.gradient(y, x)     # Will compute to 6.0
d2y_dx2 = g.gradient(dy_dx, x) 


# In[8]:


import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.9")   # please manually install torch 1.9 if Colab changes its default version

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import transforms
import cv2

#im = image.numpy()
im = cv2.imread("FLIR_08486.jpeg")
plt.imshow(im)
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml")
predictor = DefaultPredictor(cfg) # create a defaultPredictor class to predictor
outputs = predictor(im)


# In[84]:


import torch, gc

gc.collect()
torch.cuda.empty_cache()


# ### 1. Gradient tape to get derivative of loss function
# Problem: gradient is none, loss function not differentiable wrt imege input x?

# In[10]:


#In our case,0 is no object, 1 is having object, p(have object) = 1-p(not have object)
import tensorflow as tf

y_true = [[0, 0], [0, 0]]
y_pred = [[0.68, 0.51], [0.78, 0.47]]
# Using default 'auto'/'sum_over_batch_size' reduction type.
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
bce(y_true, y_pred).numpy()


centers = tf.Variable([0.05,0.3,0.35,0.6,0.67,0.9])
y_true = tf.Variable([0])
#model.trainable_variables = centers
bce = tf.keras.losses.BinaryCrossentropy()
with tf.GradientTape(watch_accessed_variables=False) as tape:
    image = tf.Variable(image)
    tape.watch(image)
    #prediction = pretrained_model(image)
    prediction = tf.constant([0.7])
    loss = bce(y_true , prediction)
    # Get the gradients of the loss w.r.t to the input image.
gradient = tape.gradient(loss, centers)
# Get the sign of the gradients to create the perturbation
signed_grad = tf.sign(gradient)
print(signed_grad)


# In[11]:


#Example 1
# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam()

# Iterate over the batches of a dataset.
for x, y in dataset:
    # Open a GradientTape.
    with tf.GradientTape() as tape:
        tape.watch(x)
        # Forward pass.
        logits = model(x)
        # Loss value for this batch.
        loss_value = loss_fn(y, logits)

    # Get gradients of loss wrt the weights.
    gradients = tape.gradient(loss_value, model.trainable_weights)

    # Update the weights of the model.
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))


# In[12]:


#Example 2
x = tf.constant(3.0)
with tf.GradientTape() as g:
    g.watch(x)
    with tf.GradientTape() as gg:
        gg.watch(x)
        y = x * x### Gradient tape to get derivative of loss function
dy_dx = gg.gradient(y, x)     # Will compute to 6.0
d2y_dx2 = g.gradient(dy_dx, x) 


# ### 2. Keras model and self define layer

# In[13]:


model_input = list()
path = 'FLIR_people_select/train/images/'
pathlist = Path(path).rglob('*.jpeg')
for path in pathlist:
    path_in_str = str(path)
    print(path_in_str)
    image_raw = tf.io.read_file(path_in_str)
    image = tf.image.decode_image(image_raw,dtype=tf.dtypes.float32)
    image = tf.reshape(image,[1,327680])
    model_input.append(image)


# In[14]:


import tensorflow as tf
from tensorflow.python.framework.ops import enable_eager_execution
enable_eager_execution()


input_shape = (512, 640, 1) #327680 #640 x 512

class MyDenseLayer(tf.keras.layers.Layer):
    #parameters
    global count
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs
    
    def build(self, input_shape):
        def initializer(*args, **kwargs):
            global count
            count = 0
            kernel =  tf.random.uniform((1,6),minval=0,maxval=1,dtype=tf.dtypes.float32)
            return kernel
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1,6),
                                      initializer=initializer,
                                      trainable=True)
        super(MyDenseLayer, self).build(input_shape)

    def call(self, image_ori): #actual action of the layer
        def input_preprocess(image_ori):
            im_nd = image.numpy()
            #im_nd = np.repeat(im_nd, 3, axis=2)
            return im_nd
        
        global count
        ind = idxs[count]
        s = 10
        sig = 5.07
        rotate_ran = 20
        im,im_r,im_b = model_patches(image_ori,ind,self.kernel,s,sig,rotate_ran)
        im,im_r,im_b = input_preprocess(im),input_preprocess(im_r),input_preprocess(im_b)
        #get expected value for different transformation
        score = yolov3_score(image) # several people, take average
        score_r = yolov3_score(im_r)
        score_b = yolov3_score(im_b)
        score_expected = np.mean(score,score_r,score_b)
        
        #total_input, number of input images
        total_input = 710
        
        score = score_expected/total_input
        #score = rcnn()
        count+=1
        return score

layer = MyDenseLayer(6)
layer.build(input_shape) # <-- example of input shape
print(layer.trainable_variables)

class MyModel(tf.keras.Model):

    def __init__(self,layer):
        super(MyModel, self).__init__()
        self.layer = layer

    def call(self, inputs):
        #inputs are multiple/single image
        score = layer(inputs)
        return score # pred_result

model = MyModel(layer)
model.compile(loss= tf.keras.losses.MeanSquaredError(), optimizer='sgd')
#model.fit(inputs), image with label 0.

model.fit(model_input,tf.constant(0, shape = (1,710)),verbose=1)


# ### 3. Keras backend

# In[15]:


import keras.backend as K

# Get the loss and gradient of the loss wrt the inputs
loss = K.binary_crossentropy(1, 0)
grads = K.gradients(loss, model.input)

loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern(input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, input_image)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
    return signed_grad

### cost implementation

import torch.nn.functional as F
input = tensor([[[[0.5546, 0.1304, 0.9288],
          [0.6879, 0.3553, 0.9984],
          [0.1474, 0.6745, 0.8948]],
         [[0.8524, 0.2278, 0.6476],
          [0.6203, 0.6977, 0.3352],
          [0.4946, 0.4613, 0.6882]]]])
target = tensor([[[1, 1, 1],
         [1, 1, 1],
         [0, 0, 1]]])

cost = F.cross_entropy(input,target)

input = torch.randn(3, 5, requires_grad=True)
target = torch.randint(5, (3,), dtype=torch.int64)
loss = F.cross_entropy(input, target)
loss.backward()

centers = [0.05,0.3,0.35,0.6,0.67,0.9]
#target = 0 #to minimize score, maximize loss

target = 

loss = K.categorical_crossentropy(, )
grads = K.gradients(loss,tensor(centers))


# In[ ]:





# In[ ]:





# In[ ]:




