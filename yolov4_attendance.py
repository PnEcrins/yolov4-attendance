#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Garel 01.12.2022

## pour détecter les erreurs dans les dossiers photos (photo qui a planté), lancer dans le dossier : 
## identify -regard-warnings *.JPG > out.txt
## identify -verbose *.png 2>&1 | grep "corrupt image"
## et chercher "error"

## temps de calculs pour 3600 photos --> 1.35 heures; 9999 photos en travaillant en même temps --> 3.3 heures

## look at :
## https://colab.research.google.com/github/sicara/tf2-yolov4/blob/master/notebooks/YoloV4_Dectection_Example.ipynb#scrollTo=4fMXYy-GG9H8

## requirement format : .h5.
## Convert .weights in .h5 by using 'convert-darknet-weights',
## python /home/mgarel/.local/bin/convert-darknet-weights /home/mgarel/Mathieu/Travail/Collaborations/PP/detection_classification_sous_R/yolo/yolov4.weights -o yolov4.h5
## you can also download it --> http://mathieu.garel.free.fr/yolo/yolov4.h5

## packages requirements
## yay python-pip
## pip install tensorflow
## pip install tf2_yolov4
## yay pandas pillow openpyxl python-matplotlib

## emacs
## python consol C-c C-p
## eval region/line C-c C-r


##################
## user defined ##
##################

## folder path with pictures
folder_pics = "../sample/20190906_20190912"

## where to find the model
folder_model = "model/yolov4.h5"

## folder path for saving results (table, annotated pictures)
folder_out = "../output/"

## threshold for classification
thresh = 0.45

###################
## load packages ##
###################

import os

import tensorflow as tf
import pandas as pd

from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4

import timeit
import time

import matplotlib.pyplot as plt


###########
## model ##
###########

HEIGHT, WIDTH = (640, 960)

model = YOLOv4(
    input_shape=(HEIGHT, WIDTH, 3),
    anchors=YOLOV4_ANCHORS,
    num_classes=80,
    training=False,
    yolo_max_boxes=100,
    yolo_iou_threshold=0.5,
    yolo_score_threshold=thresh,
)

model.load_weights(f'{folder_model}') 


##########
## data ##
##########

## based on COCO dataset and 80 classes
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]



###########
## plots ##
###########

## colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

## plot (set export_img to true if you want to export the annotated pictures)
def plot_img(pil_img, boxes, scores, classes, export_img):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    
    for (xmin, ymin, xmax, ymax), score, cl in zip(boxes.tolist(), scores.tolist(), classes.tolist()):
        if score > 0:
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=COLORS[cl % 6], linewidth=3))
            text = f'{CLASSES[cl]}: {score:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    
    if export_img:
        plt.savefig(f'{folder_out}/{qui}.png' , dpi = 80)
        # change resolution to speed up the process/save disk usage
        plt.close()

    else:
        plt.show()


###############
## run model ##
###############


res = []

export_img = False
# set to True if you want top export the annotated images (computing time = +30%)

start = timeit.default_timer() 


nb_elements = len(os.listdir(folder_pics))
print(f"Il y a {nb_elements} images à classifier")
count = 0
for qui in os.listdir(folder_pics):
    if (qui.endswith(".jpg")) or ((qui.endswith(".JPG"))) :# jpg or JPG
        count+=1
        if count%100 == 0:
            print(f"Encore {nb_elements-count} images à classifier")
        where = f"{folder_pics}/{qui}"
        image = tf.io.read_file(where)

        image = tf.image.decode_image(image)
        image = tf.image.resize(image, (HEIGHT, WIDTH))
        images = tf.expand_dims(image, axis=0) / 255.0
        
        #################
        ## predictions ##
        #################
        
        boxes, scores, classes, valid_detections = model.predict(images)

        
        ##################
        ## image export ##
        ##################
        
        if export_img:
            plot_img(images[0], boxes[0] * [WIDTH, HEIGHT, WIDTH, HEIGHT],
                     scores[0], classes[0].astype(int), export_img)

            
        ##########################
        ## export des résultats ##
        ##########################
        
        for i, j in zip(classes[0].tolist(), scores[0].tolist()):
            if j > 0:
                res.append([CLASSES[int(i)],j,where])

                
df = pd.DataFrame(res, columns=['class', 'score', 'photo']) # export results as csv
timestr = time.strftime("%Y%m%d-%H%M%S") # unique name based on date.time
df.to_csv(f'{folder_out}/yolo_{timestr}.csv')

stop = timeit.default_timer()
print('Time: ', stop - start) # get an idea of computing time
                
   
# see the last image
# plot_img(
#     images[0],
#     boxes[0] * [WIDTH, HEIGHT, WIDTH, HEIGHT],
#     scores[0],
#     classes[0].astype(int),False
# )







    
