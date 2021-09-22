cd /home/hwjin/FLIR

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

import torch.nn as nn
import keras.backend as K
from torch import tensor 

import torch.nn as nn
import keras.backend as K
from torch import tensor 


import warnings
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

#https://blog.csdn.net/qq_38048756/article/details/103208834
domain = 'train/'

### set up for yolov3 detect.pyaroot = 'FLIR_people_select/' 

jsonroot = 'FLIR_ADAS_1_3/' 
jsonfile = jsonroot + domain + 'thermal_annotations.json'
coco = COCO(jsonfile)

# path to store file
# path =

#domain = 'train/'


### Image index of input images

cd FLIR/

import pickle
with open ('FLIR_people_select_trainIdx', 'rb') as fp:
     idxs = pickle.load(fp)

### Define function

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
            patch = pat.multiple_gussians(centers,s,sig ,box_size, x ,y)
            #print(patch.shape)
            
            #patche one
            im_p = pat.patched_image(img,patch,x,y,w,h)
            im = tf.convert_to_tensor(value, dtype=None, dtype_hint=None, name=None)
            #im = Image.fromarray(np.uint8(im_p))
            #im.save(save_path + '.jpeg','JPEG')
            
            #rotated patch
            patch_rotate = pat.rand_rotate(patch,angle)
            im_r = pat.patched_image(img, patch_rotate,x,y,w,h)
            #im = Image.fromarray(np.uint8(im_p))
            #im.save(save_path + '_r.jpeg','JPEG')
            
            #bright-changed patch
            patch_bright = pat.adjust_contrast_bri(contrast,brightness,patch)
            im_b = pat.patched_image(img, patch_bright ,x,y,w,h)
            #im = Image.fromarray(np.uint8(im_p))
            #im.save(save_path + '_b.jpeg','JPEG')
            
            return im,im_r,im_b
        else:
            continue
    
    
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
    #detector2_scores = faster_rcnn_scores(centers)
    #L1_detector = 


### Import input Iamge

cd ..

import tensorflow as tf
image_raw = tf.io.read_file('FLIR_08486.jpeg')
image = tf.image.decode_image(image_raw,dtype=tf.dtypes.float32)

### Install pytorch-yolov3 api/get inference score

cd PyTorch-YOLOv3/

#image

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

### Inference rcnn

# check pytorch installation: 
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

print(torch.__version__)
print(torchvision.__version__)

#### Solve cuda out of memory

import torch, gc

gc.collect()
torch.cuda.empty_cache()

## Draft for optimization (trail of possible methods)

### 1. Gradient tape to get derivative of loss function

Problem: gradient is none, loss function not differentiable wrt imege input x?

#In our case,0 is no object, 1 is having object, p(have object) = 1-p(not have object)
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

#Example 2
x = tf.constant(3.0)
with tf.GradientTape() as g:
    g.watch(x)
    with tf.GradientTape() as gg:
        gg.watch(x)
        y = x * x### Gradient tape to get derivative of loss function
dy_dx = gg.gradient(y, x)     # Will compute to 6.0
d2y_dx2 = g.gradient(dy_dx, x) 

### 2. Keras model and self define layer

#### Input of the model, a tensor image list

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

model_input

model.trainable_variables

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

cd /home/hwjin/FLIR

what if we use model to detect image in advance save it, then import it into function
grad return none maybe it is not differentable

### 3. Keras backend

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

