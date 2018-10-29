#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle
import glob
import cv2, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from pylab import array, plot, show, axis, arange, uint8
import tensorflow as tf
from tensorflow import keras


# In[3]:


train_dir = 'TrainIJCNN2013/*.ppm'


# In[4]:


def brightness_equilization(img):
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    return cdf[img]


# In[5]:


images = []
for filename in sorted(glob.iglob(train_dir, recursive=False)):
    orig_img = cv2.imread(filename) 
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    brighter_img = brightness_equilization(orig_img)
    img_hsv = cv2.cvtColor(brighter_img, cv2.COLOR_BGR2HSV)
   
    # threshold for hue channel in red range
    red_min = np.array([10,150,50], np.uint8)
    red_max = np.array([255,255,180], np.uint8)
    threshold_red_img = cv2.inRange(img_hsv, red_min, red_max)
    threshold_red_img = cv2.cvtColor(threshold_red_img, cv2.COLOR_GRAY2RGB)
    gray_img = cv2.cvtColor(threshold_red_img, cv2.COLOR_RGB2GRAY)
    
    processed_img = gray_img/255.0
    images.append(processed_img)
    


# In[325]:


for image in images[:5]:
    print("shape: {0}, min: {1}, max: {2}".format(
          image.shape, image.min(), image.max()))


# In[320]:


# Preparing training labels
i=0
circular_labels = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
label_file  = open("TrainIJCNN2013/gt.txt", "r")
lines=label_file.readlines()
labels = np.zeros(shape = (600,1))
prev_image=""
for line in lines:
    image_details = line.split(';')
    image_name = image_details[0]
    image_type = int(image_details[5])
    if(image_name == prev_image and labels[i] ==0):
        if(image_type in circular_labels):
            labels[i] = 1
    elif(image_name == prev_image and labels[i] ==1):
        continue;
    else:
        if(image_type in circular_labels):
            labels[i] = 1
        i+=1
    prev_image = image_name


# In[318]:


train_data = np.asarray(images)


# In[321]:


train_labels = labels

