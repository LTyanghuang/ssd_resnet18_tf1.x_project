import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)
classes_names = {'aeroplane': "1",
    'bicycle': "2",
    'bird': "3",
    'boat': "4",
    'bottle': "5",
    'bus': "6",
    'car': "7",
    'cat': "8",
    'chair': "9",
    'cow': "10",
    'diningtable': "11",
    'dog': "12",
    'horse': "13",
    'motorbike': "14",
    'person': "15",
    'pottedplant': "16",
    'sheep': "17",
    'sofa': "18",
    'train': "19",
    'tvmonitor': "20"}

# Input placeholder.
net_shape = (224, 224)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _, = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = './logs/model.ckpt-298579'
# ckpt_filename = './checkpoints/ssd_300_vgg.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)
# Main image processing routine.
def process_image(img, select_threshold=0.85, nms_threshold=0.05, net_shape=(224, 224)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes
# Test on some demo image and visualize output.
# path = './demo/person.jpg'
txt_saved_dir = './datasets/detection-results/'
imgs_path = "./datasets/VOC07test/VOC2007/JPEGImages"
imgs_list = os.listdir(imgs_path)
count = 0
total_number = len(imgs_list)
for image in imgs_list:
    path = os.path.join(imgs_path, image)
    img = mpimg.imread(path)
    # print("path = ", path)
    height = img.shape[0]
    width = img.shape[1]
    rclasses, rscores, rbboxes =  process_image(img)
    # print("rclasses = ", rclasses)
    # print("rscores = ", rscores)
    # print("rbboxes = ", rbboxes)
    print("process finished {}/{}".format(count, total_number))
    count += 1
    if rclasses.shape[0] == 0:
        continue
    else:
        with open(txt_saved_dir + image[:-4] + '.txt', 'w') as f:
            for i in range(rclasses.shape[0]):
                ymin = int(rbboxes[i, 0] * height)
                xmin = int(rbboxes[i, 1] * width)
                ymax = int(rbboxes[i, 2] * height)
                xmax = int(rbboxes[i, 3] * width)
                for (k, v) in classes_names.items():
                    if str(rclasses[i]) ==  v:
                        f.write(k + " "+ str(rscores[i]) + " " + str(xmin) + " "+ str(ymin) + " " + str(xmax) + " " + str(ymax) + '\n')
        

    # visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
    # image_name = image
    # visualization.plt_bboxes(img, rclasses, rscores, rbboxes, image_name)
