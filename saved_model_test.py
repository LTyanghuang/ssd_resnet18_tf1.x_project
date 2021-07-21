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
gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (224, 224)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)
print("bbox_img = ", bbox_img)
# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _, locs, preds, ls, ps = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# # Restore SSD model.
# ckpt_filename = './logs/model.ckpt-79928'
# # ckpt_filename = './checkpoints/ssd_300_vgg.ckpt'
# isess.run(tf.global_variables_initializer())
# saver = tf.train.Saver()
# saver.restore(isess, ckpt_filename)

isess.run(tf.global_variables_initializer())
tf.saved_model.loader.load(isess, ["serve"], "./ssd_resnet18_224_1")
    


# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)
# Main image processing routine.
def process_image(img, select_threshold=0.7, nms_threshold=0.1, net_shape=(224, 224)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                            feed_dict={img_input: img})
    print("rbbox_img = ", rbbox_img)
    print("rbbox_img type = ", type(rbbox_img))
    for i in range(len(rpredictions)):
        print("rpredictions shape = ", rpredictions[i].shape)
        print("rpredictions type = ", type(rpredictions[i]))
        print("rlocalisations shape = ", rlocalisations[i].shape)
    print("rlocalisations[5] = ", rlocalisations[5])
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    print("rscores = ", rscores)
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes
# Test on some demo image and visualize output.
path = './demo/person.jpg'


img = mpimg.imread(path)
rclasses, rscores, rbboxes =  process_image(img)
print("rclasses = ", rclasses)
print("rscores = ", rscores)
# visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
