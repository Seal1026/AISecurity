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


### Use pretrained model to inference
- Single image case

im = cv2.imread("FLIR_people_select/train/images/FLIR_00001.jpeg")
plt.imshow(im)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# to solve raise URLError(err)
# urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:777)>

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml")
predictor = DefaultPredictor(cfg) # create a defaultPredictor class to predictor

outputs = predictor(im) #when the class is call=> perform inference
#input: original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
#https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-output-format

# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)
print(outputs["instances"].scores)

# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
plt.imshow(out.get_image()[:, :, ::-1])

### Add a loop to infer all images
- multiple image case, use same predictor as above

#### Save results: save scores of all detected people => prepare for loss function 
save results:https://github.com/facebookresearch/detectron2/issues/858
full tutorials: https://towardsdatascience.com/object-detection-in-6-steps-using-detectron2-705b92575578

detect_score = list()
for idx in range(710):
    im = cv2.imread("FLIR_people_select/train/images/FLIR_{:05d}.jpeg".format(idx+1))
    outputs = predictor(im)
    scores = outputs["instances"].scores.tolist()
    pred_classes = outputs["instances"].pred_classes.tolist()
    for i,pre_class in enumerate(pred_classes):
        if pre_class == 0:
            print(scores[i])
            detect_score.append(scores[i])                 

detect_score 

### Detectron2 Model Zoo and Baselines/model selections

https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md

--config-file,--opt MODEL.WEIGHTS sould be consistent 

### Demo

!python demo/demo.py --config-file configs/COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml  --input FLIR_08487.jpeg --output output_FLIR/ \--opts MODEL.WEIGHTS weights_FLIR/COCO-Detection/R50-1x/model_final_721ade.pkl 
    #the weights cannot be download from url due to certificate ssl restrction
    # use https://detectron2.readthedocs.io/en/latest/tutorials/configs.html?highlight=opts#basic-usage to load weight

!python demo/demo.py --input FLIR_08487.jpeg --output output_FLIR/ \--opts MODEL.WEIGHTS weights_FLIR/model_final_f10217.pkl 

!python demo/demo.py -h

#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context
#!pip install python-certifi-win32 --trusted-host files.pythonhosted.org --trusted-host pypi.org --trusted-host pypi.python.org
#import urllib3
#urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
#import requests
#requests.get('https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl', verify=False)
#url = 'https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'
#r = requests.post(url, verify='certificate.cer')
