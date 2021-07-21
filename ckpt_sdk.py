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

# Input placeholder.
net_shape = (224, 224)
data_format = 'NHWC'
'''
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)
print("image_4d shape = ", image_4d.shape)
print("image_4d type = ", type(image_4d))

inputs_img = tf.placeholder(shape=[1, 300, 300, 3], dtype=tf.float32, name='input_image')
print("inputs_img shape = ", inputs_img.shape)
print("inputs_img type = ", type(inputs_img))
# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _, locs, preds = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = './logs/model.ckpt-113991'
# ckpt_filename = './checkpoints/ssd_300_vgg.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)
'''


# SSD default anchor boxes.
# ssd_anchors = ssd_net.anchors(net_shape)
# Main image processing routine.
def process_image(img, select_threshold=0.8, nms_threshold=0.2, net_shape=(224, 224)):
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

inputs_img = tf.placeholder(shape=[1, 224, 224, 3], dtype=tf.float32, name='input_image')
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format='NHWC')):
    predictions, localisations, _, _, = ssd_net.net(inputs_img, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = './logs/model.ckpt-377829'
# ckpt_filename = './checkpoints/ssd_300_vgg.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

print("predictions[0] type = ", type(predictions[0]))
print("predictions[0] shape = ", predictions[0].shape)


outs = {'ls_0':localisations[0], 'ps_0':predictions[0], 'ls_1':localisations[1], 'ps_1':predictions[1],
       'ls_2':localisations[2], 'ps_2':predictions[2], 'ls_3':localisations[3], 'ps_3':predictions[3],
       'ls_4':localisations[4], 'ps_4':predictions[4], 'ls_5':localisations[5], 'ps_5':predictions[5]}

tf.saved_model.simple_save(isess, "./ssd_resnet18_224",inputs={'input_image':inputs_img},
            outputs={'ls_0':outs['ls_0'], 'ps_0':outs['ps_0'],'ls_1':outs['ls_1'], 'ps_1':outs['ps_1'],
                    'ls_2':outs['ls_2'], 'ps_2':outs['ps_2'],'ls_3':outs['ls_3'], 'ps_3':outs['ps_3'],
                    'ls_4':outs['ls_4'], 'ps_4':outs['ps_4'],'ls_5':outs['ls_5'], 'ps_5':outs['ps_5']}) 
print("saved model end ........")

'''
################ saved model ##########################
# inputs_img = tf.placeholder(shape=[1, 300, 300, 3], dtype=tf.float32, name='input_image')
print("localisations = ", localisations)
print("localisations type = ", type(localisations[0]))
outs = {'loc_outs':locs, 'p_outs':preds}
tf.saved_model.simple_save(isess, "./ssd_resnet18_224_0",inputs={'input_image':inputs_img},outputs={'loc_outs':outs['loc_outs'], 'p_outs':outs['p_outs']}) 
print("saved model end ........")
################ saved model ##########################
'''
def process_image1(img, select_threshold=0.8, nms_threshold=0.2, net_shape=(224, 224)):
    print("input start .....")
    
    
    
# Test on some demo image and visualize output.
path = './demo/person.jpg'


img = mpimg.imread(path)
img1 = cv2.resize(img,(224,224))
img1 = np.array(img1, dtype=float)
img1 = np.expand_dims(img1, 0)
print("img1 shape = ", img1.shape)
process_image1(img1)
# rclasses, rscores, rbboxes =  process_image(img)
# print("rclasses = ", rclasses)
# print("rscores = ", rscores)
# # visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
# visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
