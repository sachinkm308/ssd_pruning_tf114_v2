#!/usr/bin/env python
# coding: utf-8

# # SSD7 Training Tutorial
# 
# This tutorial explains how to train an SSD7 on the Udacity road traffic datasets, and just generally how to use this SSD implementation.
# 
# Disclaimer about SSD7:
# As you will see below, training SSD7 on the aforementioned datasets yields alright results, but I'd like to emphasize that SSD7 is not a carefully optimized network architecture. The idea was just to build a low-complexity network that is fast (roughly 127 FPS or more than 3 times as fast as SSD300 on a GTX 1070) for testing purposes. Would slightly different anchor box scaling factors or a slightly different number of filters in individual convolution layers make SSD7 significantly better at similar complexity? I don't know, I haven't tried.

# In[1]:


from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil

import keras
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

#from models.keras_ssd7 import build_model
from models.keras_ssd7_copy import build_model  #Creating model that can be used for pruning

from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation

import os.path
from os import path


get_ipython().run_line_magic('matplotlib', 'inline')


# ### Imports for the pruning

# In[ ]:


import tempfile
import os

import tensorflow as tf
import numpy as np

from tensorflow import keras

get_ipython().run_line_magic('load_ext', 'tensorboard')


# ## 1. Set the model configuration parameters
# 
# The cell below sets a number of parameters that define the model configuration. The parameters set here are being used both by the `build_model()` function that builds the model as well as further down by the constructor for the `SSDInputEncoder` object that is needed to to match ground truth and anchor boxes during the training.
# 
# Here are just some comments on a few of the parameters, read the documentation for more details:
# 
# * Set the height, width, and number of color channels to whatever you want the model to accept as image input. If your input images have a different size than you define as the model input here, or if your images have non-uniform size, then you must use the data generator's image transformations (resizing and/or cropping) so that your images end up having the required input size before they are fed to the model. to convert your images to the model input size during training. The SSD300 training tutorial uses the same image pre-processing and data augmentation as the original Caffe implementation, so take a look at that to see one possibility of how to deal with non-uniform-size images.
# * The number of classes is the number of positive classes in your dataset, e.g. 20 for Pascal VOC or 80 for MS COCO. Class ID 0 must always be reserved for the background class, i.e. your positive classes must have positive integers as their IDs in your dataset.
# * The `mode` argument in the `build_model()` function determines whether the model will be built with or without a `DecodeDetections` layer as its last layer. In 'training' mode, the model outputs the raw prediction tensor, while in 'inference' and 'inference_fast' modes, the raw predictions are being decoded into absolute coordinates and filtered via confidence thresholding, non-maximum suppression, and top-k filtering. The difference between latter two modes is that 'inference' uses the decoding procedure of the original Caffe implementation, while 'inference_fast' uses a faster, but possibly less accurate decoding procedure.
# * The reason why the list of scaling factors has 5 elements even though there are only 4 predictor layers in tSSD7 is that the last scaling factor is used for the second aspect-ratio-1 box of the last predictor layer. Refer to the documentation for details.
# * `build_model()` and `SSDInputEncoder` have two arguments for the anchor box aspect ratios: `aspect_ratios_global` and `aspect_ratios_per_layer`. You can use either of the two, you don't need to set both. If you use `aspect_ratios_global`, then you pass one list of aspect ratios and these aspect ratios will be used for all predictor layers. Every aspect ratio you want to include must be listed once and only once. If you use `aspect_ratios_per_layer`, then you pass a nested list containing lists of aspect ratios for each individual predictor layer. This is what the SSD300 training tutorial does. It's your design choice whether all predictor layers should use the same aspect ratios or whether you think that for your dataset, certain aspect ratios are only necessary for some predictor layers but not for others. Of course more aspect ratios means more predicted boxes, which in turn means increased computational complexity.
# * If `two_boxes_for_ar1 == True`, then each predictor layer will predict two boxes with aspect ratio one, one a bit smaller, the other one a bit larger.
# * If `clip_boxes == True`, then the anchor boxes will be clipped so that they lie entirely within the image boundaries. It is recommended not to clip the boxes. The anchor boxes form the reference frame for the localization prediction. This reference frame should be the same at every spatial position.
# * In the matching process during the training, the anchor box offsets are being divided by the variances. Leaving them at 1.0 for each of the four box coordinates means that they have no effect. Setting them to less than 1.0 spreads the imagined anchor box offset distribution for the respective box coordinate.
# * `normalize_coords` converts all coordinates from absolute coordinate to coordinates that are relative to the image height and width. This setting has no effect on the outcome of the training.

# In[ ]:


img_height = 300 # Height of the input images
img_width = 480 # Width of the input images
img_channels = 3 # Number of color channels of the input images
intensity_mean = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 5 # Number of positive classes
scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size


# ## 2. Build or load the model
# 
# You will want to execute either of the two code cells in the subsequent two sub-sections, not both.

# ### 2.1 Create a new model
# 
# If you want to create a new model, this is the relevant section for you. If you want to load a previously saved model, skip ahead to section 2.2.
# 
# The code cell below does the following things:
# 1. It calls the function `build_model()` to build the model.
# 2. It optionally loads some weights into the model.
# 3. It then compiles the model for the training. In order to do so, we're defining an optimizer (Adam) and a loss function (SSDLoss) to be passed to the `compile()` method.
# 
# `SSDLoss` is a custom Keras loss function that implements the multi-task log loss for classification and smooth L1 loss for localization. `neg_pos_ratio` and `alpha` are set as in the paper.

# In[ ]:


# 1: Build the Keras model

K.clear_session() # Clear previous models from memory.

model = build_model(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_global=aspect_ratios,
                    aspect_ratios_per_layer=None,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=intensity_mean,
                    divide_by_stddev=intensity_range)

# 2: Optional: Load some weights

#model.load_weights('./ssd7_weights.h5', by_name=True)

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss, metrics=[tf.keras.metrics.Accuracy()])


# ### Defining the model for pruning -> #pruning

# In[17]:


from __future__ import division
import numpy as np
#from keras.models import Model
#from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation
from keras.regularizers import l2
import keras.backend as K

from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

K.clear_session() # Clear previous models from memory.

img_height = 300 # Height of the input images
img_width = 480 # Width of the input images
img_channels = 3 # Number of color channels of the input images
intensity_mean = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 5 # Number of positive classes
scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size


l2_regularization=0.0
image_size=(img_height, img_width, img_channels)
aspect_ratios_global=[0.5, 1.0, 2.0]
aspect_ratios_per_layer=None
min_scale=0.1
max_scale=0.9
subtract_mean=None
divide_by_stddev=None
swap_channels=False
confidence_thresh=0.01
iou_threshold=0.45
top_k=200
nms_max_output_size=400
return_predictor_sizes=False
coords='centroids'
mode='training'


n_predictor_layers = 4 # The number of predictor conv layers in the network
n_classes += 1 # Account for the background class.
l2_reg = l2_regularization # Make the internal name shorter.
img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]


############################################################################
# Get a few exceptions out of the way.
############################################################################

if aspect_ratios_global is None and aspect_ratios_per_layer is None:
    raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
if aspect_ratios_per_layer:
    if len(aspect_ratios_per_layer) != n_predictor_layers:
        raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

if (min_scale is None or max_scale is None) and scales is None:
    raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
if scales:
    if len(scales) != n_predictor_layers+1:
        raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
else: # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
    scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

if len(variances) != 4: # We need one variance value for each of the four box coordinates
    raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
variances = np.array(variances)
if np.any(variances <= 0):
    raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

if (not (steps is None)) and (len(steps) != n_predictor_layers):
    raise ValueError("You must provide at least one step value per predictor layer.")

if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
    raise ValueError("You must provide at least one offset value per predictor layer.")

############################################################################
# Compute the anchor box parameters.
############################################################################

# Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
if aspect_ratios_per_layer:
    aspect_ratios = aspect_ratios_per_layer
else:
    aspect_ratios = [aspect_ratios_global] * n_predictor_layers

# Compute the number of boxes to be predicted per cell for each predictor layer.
# We need this so that we know how many channels the predictor layers need to have.
if aspect_ratios_per_layer:
    n_boxes = []
    for ar in aspect_ratios_per_layer:
        if (1 in ar) & two_boxes_for_ar1:
            n_boxes.append(len(ar) + 1) # +1 for the second box for aspect ratio 1
        else:
            n_boxes.append(len(ar))
else: # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
    if (1 in aspect_ratios_global) & two_boxes_for_ar1:
        n_boxes = len(aspect_ratios_global) + 1
    else:
        n_boxes = len(aspect_ratios_global)
    n_boxes = [n_boxes] * n_predictor_layers

if steps is None:
    steps = [None] * n_predictor_layers
if offsets is None:
    offsets = [None] * n_predictor_layers

############################################################################
# Define functions for the Lambda layers below.
############################################################################

def identity_layer(tensor):
    return tensor

def input_mean_normalization(tensor):
    return tensor - np.array(subtract_mean)

def input_stddev_normalization(tensor):
    return tensor / np.array(divide_by_stddev)

def input_channel_swap(tensor):
    if len(swap_channels) == 3:
        return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)
    elif len(swap_channels) == 4:
        return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]], tensor[...,swap_channels[3]]], axis=-1)

############################################################################
# Build the network.
############################################################################

x = keras.layers.Input(shape=(img_height, img_width, img_channels))

# The following identity layer is only needed so that the subsequent lambda layers can be optional.
x1 = keras.layers.Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
if not (subtract_mean is None):
    x1 = keras.layers.Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x1)
if not (divide_by_stddev is None):
    x1 = keras.layers.Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)
if swap_channels:
    x1 = keras.layers.Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(x1)

conv1 = keras.layers.Conv2D(32, (5, 5), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1')(x1)
conv1 = keras.layers.BatchNormalization(axis=3, momentum=0.99, name='bn1')(conv1) # Tensorflow uses filter format [filter_height, filter_width, in_channels, out_channels], hence axis = 3
conv1 = keras.layers.ELU(name='elu1')(conv1)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)

conv2 = keras.layers.Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2')(pool1)
conv2 = keras.layers.BatchNormalization(axis=3, momentum=0.99, name='bn2')(conv2)
conv2 = keras.layers.ELU(name='elu2')(conv2)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)

conv3 = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3')(pool2)
conv3 = keras.layers.BatchNormalization(axis=3, momentum=0.99, name='bn3')(conv3)
conv3 = keras.layers.ELU(name='elu3')(conv3)
pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)

conv4 = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4')(pool3)
conv4 = keras.layers.BatchNormalization(axis=3, momentum=0.99, name='bn4')(conv4)
conv4 = keras.layers.ELU(name='elu4')(conv4)
pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)

conv5 = keras.layers.Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5')(pool4)
conv5 = keras.layers.BatchNormalization(axis=3, momentum=0.99, name='bn5')(conv5)
conv5 = keras.layers.ELU(name='elu5')(conv5)
pool5 = keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool5')(conv5)

conv6 = keras.layers.Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6')(pool5)
conv6 = keras.layers.BatchNormalization(axis=3, momentum=0.99, name='bn6')(conv6)
conv6 = keras.layers.ELU(name='elu6')(conv6)
pool6 = keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool6')(conv6)

conv7 = keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7')(pool6)
conv7 = keras.layers.BatchNormalization(axis=3, momentum=0.99, name='bn7')(conv7)
conv7 = keras.layers.ELU(name='elu7')(conv7)
'''
# The next part is to add the convolutional predictor layers on top of the base network
# that we defined above. Note that I use the term "base network" differently than the paper does.
# To me, the base network is everything that is not convolutional predictor layers or anchor
# box layers. In this case we'll have four predictor layers, but of course you could
# easily rewrite this into an arbitrarily deep base network and add an arbitrary number of
# predictor layers on top of the base network by simply following the pattern shown here.

# Build the convolutional predictor layers on top of conv layers 4, 5, 6, and 7.
# We build two predictor layers on top of each of these layers: One for class prediction (classification), one for box coordinate prediction (localization)
# We precidt `n_classes` confidence values for each box, hence the `classes` predictors have depth `n_boxes * n_classes`
# We predict 4 box coordinates for each box, hence the `boxes` predictors have depth `n_boxes * 4`
# Output shape of `classes`: `(batch, height, width, n_boxes * n_classes)`
classes4 = keras.layers.Conv2D(n_boxes[0] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes4')(conv4)
classes5 = keras.layers.Conv2D(n_boxes[1] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes5')(conv5)
classes6 = keras.layers.Conv2D(n_boxes[2] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes6')(conv6)
classes7 = keras.layers.Conv2D(n_boxes[3] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes7')(conv7)
# Output shape of `boxes`: `(batch, height, width, n_boxes * 4)`
boxes4 = keras.layers.Conv2D(n_boxes[0] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes4')(conv4)
boxes5 = keras.layers.Conv2D(n_boxes[1] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes5')(conv5)
boxes6 = keras.layers.Conv2D(n_boxes[2] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes6')(conv6)
boxes7 = keras.layers.Conv2D(n_boxes[3] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes7')(conv7)

# Generate the anchor boxes
# Output shape of `anchors`: `(batch, height, width, n_boxes, 8)`
anchors4 = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                       two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0],
                       clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors4')(boxes4)
anchors5 = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                       two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1],
                       clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors5')(boxes5)
anchors6 = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                       two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2],
                       clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors6')(boxes6)
anchors7 = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                       two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3],
                       clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors7')(boxes7)

# Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
# We want the classes isolated in the last axis to perform softmax on them
classes4_reshaped = keras.layers.Reshape((-1, n_classes), name='classes4_reshape')(classes4)
classes5_reshaped = keras.layers.Reshape((-1, n_classes), name='classes5_reshape')(classes5)
classes6_reshaped = keras.layers.Reshape((-1, n_classes), name='classes6_reshape')(classes6)
classes7_reshaped = keras.layers.Reshape((-1, n_classes), name='classes7_reshape')(classes7)
# Reshape the box coordinate predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
# We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
boxes4_reshaped = keras.layers.Reshape((-1, 4), name='boxes4_reshape')(boxes4)
boxes5_reshaped = keras.layers.Reshape((-1, 4), name='boxes5_reshape')(boxes5)
boxes6_reshaped = keras.layers.Reshape((-1, 4), name='boxes6_reshape')(boxes6)
boxes7_reshaped = keras.layers.Reshape((-1, 4), name='boxes7_reshape')(boxes7)
# Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
anchors4_reshaped = keras.layers.Reshape((-1, 8), name='anchors4_reshape')(anchors4)
anchors5_reshaped = keras.layers.Reshape((-1, 8), name='anchors5_reshape')(anchors5)
anchors6_reshaped = keras.layers.Reshape((-1, 8), name='anchors6_reshape')(anchors6)
anchors7_reshaped = keras.layers.Reshape((-1, 8), name='anchors7_reshape')(anchors7)

# Concatenate the predictions from the different layers and the assosciated anchor box tensors
# Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
# so we want to concatenate along axis 1
# Output shape of `classes_concat`: (batch, n_boxes_total, n_classes)
classes_concat = keras.layers.Concatenate(axis=1, name='classes_concat')([classes4_reshaped,
                                                             classes5_reshaped,
                                                             classes6_reshaped,
                                                             classes7_reshaped])

# Output shape of `boxes_concat`: (batch, n_boxes_total, 4)
boxes_concat = keras.layers.Concatenate(axis=1, name='boxes_concat')([boxes4_reshaped,
                                                         boxes5_reshaped,
                                                         boxes6_reshaped,
                                                         boxes7_reshaped])

# Output shape of `anchors_concat`: (batch, n_boxes_total, 8)
anchors_concat = keras.layers.Concatenate(axis=1, name='anchors_concat')([anchors4_reshaped,
                                                             anchors5_reshaped,
                                                             anchors6_reshaped,
                                                             anchors7_reshaped])

# The box coordinate predictions will go into the loss function just the way they are,
# but for the class predictions, we'll apply a softmax activation layer first
classes_softmax = keras.layers.Activation('softmax', name='classes_softmax')(classes_concat)

# Concatenate the class and box coordinate predictions and the anchors to one large predictions tensor
# Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
predictions = keras.layers.Concatenate(axis=2, name='predictions')([classes_softmax, boxes_concat, anchors_concat])
'''    
#print("Value of x ->", x)
print("Output  ->", predictions)

if mode == 'training':
    model = keras.models.Model(inputs=x, outputs=conv7)
    model.summary()
elif mode == 'inference':
    decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                           iou_threshold=iou_threshold,
                                           top_k=top_k,
                                           nms_max_output_size=nms_max_output_size,
                                           coords=coords,
                                           normalize_coords=normalize_coords,
                                           img_height=img_height,
                                           img_width=img_width,
                                           name='decoded_predictions')(predictions)
    model = keras.models.Model(inputs=x, outputs=decoded_predictions)
elif mode == 'inference_fast':
    decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
    model = keras.models.Model(inputs=x, outputs=decoded_predictions)
    model.summary()
else:
    raise ValueError("`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

if return_predictor_sizes:
    # The spatial dimensions are the same for the `classes` and `boxes` predictor layers.
    predictor_sizes = np.array([classes4._keras_shape[1:3],
                                classes5._keras_shape[1:3],
                                classes6._keras_shape[1:3],
                                classes7._keras_shape[1:3]])
    model, predictor_sizes
else:
    model
    


# ### Defining the model for pruning -> #pruning --> A small working part

# In[14]:


from __future__ import division
import numpy as np
#from keras.models import Model
#from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation
from keras.regularizers import l2
import keras.backend as K

from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

K.clear_session() # Clear previous models from memory.

img_height = 300 # Height of the input images
img_width = 480 # Width of the input images
img_channels = 3 # Number of color channels of the input images
intensity_mean = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 5 # Number of positive classes
scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size


l2_regularization=0.0
image_size=(img_height, img_width, img_channels)
aspect_ratios_global=[0.5, 1.0, 2.0]
aspect_ratios_per_layer=None
min_scale=0.1
max_scale=0.9
subtract_mean=None
divide_by_stddev=None
swap_channels=False
confidence_thresh=0.01
iou_threshold=0.45
top_k=200
nms_max_output_size=400
return_predictor_sizes=False
coords='centroids'
mode='training'


n_predictor_layers = 4 # The number of predictor conv layers in the network
n_classes += 1 # Account for the background class.
l2_reg = l2_regularization # Make the internal name shorter.
img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]


############################################################################
# Get a few exceptions out of the way.
############################################################################

if aspect_ratios_global is None and aspect_ratios_per_layer is None:
    raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
if aspect_ratios_per_layer:
    if len(aspect_ratios_per_layer) != n_predictor_layers:
        raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

if (min_scale is None or max_scale is None) and scales is None:
    raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
if scales:
    if len(scales) != n_predictor_layers+1:
        raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
else: # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
    scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

if len(variances) != 4: # We need one variance value for each of the four box coordinates
    raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
variances = np.array(variances)
if np.any(variances <= 0):
    raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

if (not (steps is None)) and (len(steps) != n_predictor_layers):
    raise ValueError("You must provide at least one step value per predictor layer.")

if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
    raise ValueError("You must provide at least one offset value per predictor layer.")

############################################################################
# Compute the anchor box parameters.
############################################################################

# Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
if aspect_ratios_per_layer:
    aspect_ratios = aspect_ratios_per_layer
else:
    aspect_ratios = [aspect_ratios_global] * n_predictor_layers

# Compute the number of boxes to be predicted per cell for each predictor layer.
# We need this so that we know how many channels the predictor layers need to have.
if aspect_ratios_per_layer:
    n_boxes = []
    for ar in aspect_ratios_per_layer:
        if (1 in ar) & two_boxes_for_ar1:
            n_boxes.append(len(ar) + 1) # +1 for the second box for aspect ratio 1
        else:
            n_boxes.append(len(ar))
else: # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
    if (1 in aspect_ratios_global) & two_boxes_for_ar1:
        n_boxes = len(aspect_ratios_global) + 1
    else:
        n_boxes = len(aspect_ratios_global)
    n_boxes = [n_boxes] * n_predictor_layers

if steps is None:
    steps = [None] * n_predictor_layers
if offsets is None:
    offsets = [None] * n_predictor_layers

############################################################################
# Define functions for the Lambda layers below.
############################################################################

def identity_layer(tensor):
    return tensor

def input_mean_normalization(tensor):
    return tensor - np.array(subtract_mean)

def input_stddev_normalization(tensor):
    return tensor / np.array(divide_by_stddev)

def input_channel_swap(tensor):
    if len(swap_channels) == 3:
        return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)
    elif len(swap_channels) == 4:
        return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]], tensor[...,swap_channels[3]]], axis=-1)

############################################################################
# Build the network.
############################################################################

x = keras.layers.Input(shape=(img_height, img_width, img_channels))

# The following identity layer is only needed so that the subsequent lambda layers can be optional.
x1 = keras.layers.Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
if not (subtract_mean is None):
    x1 = keras.layers.Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x1)
if not (divide_by_stddev is None):
    x1 = keras.layers.Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)
if swap_channels:
    x1 = keras.layers.Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(x1)

conv1 = keras.layers.Conv2D(32, (5, 5), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1')(x1)
conv1 = keras.layers.BatchNormalization(axis=3, momentum=0.99, name='bn1')(conv1) # Tensorflow uses filter format [filter_height, filter_width, in_channels, out_channels], hence axis = 3
conv1 = keras.layers.ELU(name='elu1')(conv1)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)


if mode == 'training':
    model = keras.models.Model(inputs=x, outputs=pool1)
    model.summary()


# ### Defining the pruning model

# In[19]:


#model.summary()

import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

epochs = 2
num_images = 18000
end_step = np.ceil(num_images / 8).astype(np.int32) * epochs
print("End_step -> ",end_step)

# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_for_pruning.summary()


# In[ ]:





# ### 2.2 Load a saved model
# 
# If you have previously created and saved a model and would now like to load it, simply execute the next code cell. The only thing you need to do is to set the path to the saved model HDF5 file that you would like to load.
# 
# The SSD model contains custom objects: Neither the loss function, nor the anchor box or detection decoding layer types are contained in the Keras core library, so we need to provide them to the model loader.
# 
# This next code cell assumes that you want to load a model that was created in 'training' mode. If you want to load a model that was created in 'inference' or 'inference_fast' mode, you'll have to add the `DecodeDetections` or `DecodeDetectionsFast` layer type to the `custom_objects` dictionary below.

# In[ ]:


path.exists('./ssd7_epoch-16_loss-2.3440_val_loss-2.4237.h5')


# In[ ]:


pwd


# In[ ]:


# TODO: Set the path to the `.h5` file of the model to be loaded.
model_path = './ssd7_epoch-16_loss-2.3440_val_loss-2.4237.h5'

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

K.clear_session() # Clear previous models from memory.

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'compute_loss': ssd_loss.compute_loss})


# ## 3. Set up the data generators for the training
# 
# The code cells below set up data generators for the training and validation datasets to train the model. You will have to set the file paths to your dataset. Depending on the annotations format of your dataset, you might also have to switch from the CSV parser to the XML or JSON parser, or you might have to write a new parser method in the `DataGenerator` class that can handle whatever format your annotations are in. The [README](https://github.com/pierluigiferrari/ssd_keras/blob/master/README.md) of this repository provides a summary of the design of the `DataGenerator`, which should help you in case you need to write a new parser or adapt one of the existing parsers to your needs.
# 
# Note that the generator provides two options to speed up the training. By default, it loads the individual images for a batch from disk. This has two disadvantages. First, for compressed image formats like JPG, this is a huge computational waste, because every image needs to be decompressed again and again every time it is being loaded. Second, the images on disk are likely not stored in a contiguous block of memory, which may also slow down the loading process. The first option that `DataGenerator` provides to deal with this is to load the entire dataset into memory, which reduces the access time for any image to a negligible amount, but of course this is only an option if you have enough free memory to hold the whole dataset. As a second option, `DataGenerator` provides the possibility to convert the dataset into a single HDF5 file. This HDF5 file stores the images as uncompressed arrays in a contiguous block of memory, which dramatically speeds up the loading time. It's not as good as having the images in memory, but it's a lot better than the default option of loading them from their compressed JPG state every time they are needed. Of course such an HDF5 dataset may require significantly more disk space than the compressed images. You can later load these HDF5 datasets directly in the constructor.
# 
# Set the batch size to to your preference and to what your GPU memory allows, it's not the most important hyperparameter. The Caffe implementation uses a batch size of 32, but smaller batch sizes work fine, too.
# 
# The `DataGenerator` itself is fairly generic. I doesn't contain any data augmentation or bounding box encoding logic. Instead, you pass a list of image transformations and an encoder for the bounding boxes in the `transformations` and `label_encoder` arguments of the data generator's `generate()` method, and the data generator will then apply those given transformations and the encoding to the data. Everything here is preset already, but if you'd like to learn more about the data generator and its data augmentation capabilities, take a look at the detailed tutorial in [this](https://github.com/pierluigiferrari/data_generator_object_detection_2d) repository.
# 
# The image processing chain defined further down in the object named `data_augmentation_chain` is just one possibility of what a data augmentation pipeline for unform-size images could look like. Feel free to put together other image processing chains, you can use the `DataAugmentationConstantInputSize` class as a template. Or you could use the original SSD data augmentation pipeline by instantiting an `SSDDataAugmentation` object and passing that to the generator instead. This procedure is not exactly efficient, but it evidently produces good results on multiple datasets.
# 
# An `SSDInputEncoder` object, `ssd_input_encoder`, is passed to both the training and validation generators. As explained above, it matches the ground truth labels to the model's anchor boxes and encodes the box coordinates into the format that the model needs.

# ### Note:
# 
# The example setup below was used to train SSD7 on two road traffic datasets released by [Udacity](https://github.com/udacity/self-driving-car/tree/master/annotations) with around 20,000 images in total and 5 object classes (car, truck, pedestrian, bicyclist, traffic light), although the vast majority of the objects are cars. The original datasets have a constant image size of 1200x1920 RGB. I consolidated the two datasets, removed a few bad samples (although there are probably many more), and resized the images to 300x480 RGB, i.e. to one sixteenth of the original image size. In case you'd like to train a model on the same dataset, you can download the consolidated and resized dataset I used [here](https://drive.google.com/open?id=1tfBFavijh4UTG4cGqIKwhcklLXUDuY0D) (about 900 MB).

# In[ ]:


# 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

# Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.

train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path= None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path= None)

# 2: Parse the image and label lists for the training and validation datasets.

# TODO: Set the paths to your dataset here.

# Images
images_dir = '../../ssd_pruning/udacity_driving_datasets/'

# Ground truth
train_labels_filename = '../../ssd_pruning/udacity_driving_datasets/labels_train_car.csv'
val_labels_filename   = '../../ssd_pruning/udacity_driving_datasets/labels_val_car.csv'

train_dataset.parse_csv(images_dir=images_dir,
                        labels_filename=train_labels_filename,
                        input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'], # This is the order of the first six columns in the CSV file that contains the labels for your dataset. If your labels are in XML format, maybe the XML parser will be helpful, check the documentation.
                        include_classes='all')

val_dataset.parse_csv(images_dir=images_dir,
                      labels_filename=val_labels_filename,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')

# Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will
# speed up the training. Doing this is not relevant in case you activated the `load_images_into_memory`
# option in the constructor, because in that cas the images are in memory already anyway. If you don't
# want to create HDF5 datasets, comment out the subsequent two function calls.

#train_dataset.create_hdf5_dataset(file_path='dataset_udacity_traffic_train.h5',
#                                  resize=False,
#                                  variable_image_size=True,
#                                  verbose=True)

#val_dataset.create_hdf5_dataset(file_path='dataset_udacity_traffic_val.h5',
#                                resize=False,
#                                variable_image_size=True,
#                                verbose=True)

#train_dataset.load_hdf5_dataset(verbose=True)
#val_dataset.load_hdf5_dataset(verbose=True)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))


# In[ ]:


# 3: Set the batch size.

batch_size = 8

# 4: Define the image processing chain.

data_augmentation_chain = DataAugmentationConstantInputSize(random_brightness=(-48, 48, 0.5),
                                                            random_contrast=(0.5, 1.8, 0.5),
                                                            random_saturation=(0.5, 1.8, 0.5),
                                                            random_hue=(18, 0.5),
                                                            random_flip=0.5,
                                                            random_translate=((0.03,0.5), (0.03,0.5), 0.5),
                                                            random_scale=(0.5, 2.0, 0.5),
                                                            n_trials_max=3,
                                                            clip_boxes=True,
                                                            overlap_criterion='area',
                                                            bounds_box_filter=(0.3, 1.0),
                                                            bounds_validator=(0.5, 1.0),
                                                            n_boxes_min=1,
                                                            background=(0,0,0))

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('classes4').output_shape[1:3],
                   model.get_layer('classes5').output_shape[1:3],
                   model.get_layer('classes6').output_shape[1:3],
                   model.get_layer('classes7').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_global=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.3,
                                    normalize_coords=normalize_coords)

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[data_augmentation_chain],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)


# ## 4. Set the remaining training parameters and train the model
# 
# We've already chosen an optimizer and a learning rate and set the batch size above, now let's set the remaining training parameters.
# 
# I'll set a few Keras callbacks below, one for early stopping, one to reduce the learning rate if the training stagnates, one to save the best models during the training, and one to continuously stream the training history to a CSV file after every epoch. Logging to a CSV file makes sense, because if we didn't do that, in case the training terminates with an exception at some point or if the kernel of this Jupyter notebook dies for some reason or anything like that happens, we would lose the entire history for the trained epochs. Feel free to add more callbacks if you want TensorBoard summaries or whatever.

# In[ ]:


# Define model callbacks.

# TODO: Set the filepath under which you want to save the weights.
model_checkpoint = ModelCheckpoint(filepath='ssd7_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)

csv_logger = CSVLogger(filename='ssd7_training_log.csv',
                       separator=',',
                       append=True)

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0.0,
                               patience=10,
                               verbose=1)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.2,
                                         patience=8,
                                         verbose=1,
                                         epsilon=0.001,
                                         cooldown=0,
                                         min_lr=0.00001)

callbacks = [model_checkpoint,
             csv_logger,
             early_stopping,
             reduce_learning_rate]


# I'll set one epoch to consist of 1,000 training steps I'll arbitrarily set the number of epochs to 20 here. This does not imply that 20,000 training steps is the right number. Depending on the model, the dataset, the learning rate, etc. you might have to train much longer to achieve convergence, or maybe less.
# 
# Instead of trying to train a model to convergence in one go, you might want to train only for a few epochs at a time.
# 
# In order to only run a partial training and resume smoothly later on, there are a few things you should note:
# 1. Always load the full model if you can, rather than building a new model and loading previously saved weights into it. Optimizers like SGD or Adam keep running averages of past gradient moments internally. If you always save and load full models when resuming a training, then the state of the optimizer is maintained and the training picks up exactly where it left off. If you build a new model and load weights into it, the optimizer is being initialized from scratch, which, especially in the case of Adam, leads to small but unnecessary setbacks every time you resume the training with previously saved weights.
# 2. You should tell `fit_generator()` which epoch to start from, otherwise it will start with epoch 0 every time you resume the training. Set `initial_epoch` to be the next epoch of your training. Note that this parameter is zero-based, i.e. the first epoch is epoch 0. If you had trained for 10 epochs previously and now you'd want to resume the training from there, you'd set `initial_epoch = 10` (since epoch 10 is the eleventh epoch). Furthermore, set `final_epoch` to the last epoch you want to run. To stick with the previous example, if you had trained for 10 epochs previously and now you'd want to train for another 10 epochs, you'd set `initial_epoch = 10` and `final_epoch = 20`.
# 3. Callbacks like `ModelCheckpoint` or `ReduceLROnPlateau` are stateful, so you might want ot save their state somehow if you want to pick up a training exactly where you left off.

# In[ ]:


# TODO: Set the epochs to train for.
# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch   = 0
final_epoch     = 20
steps_per_epoch = 1000

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size/batch_size),
                              initial_epoch=initial_epoch)


# Let's look at how the training and validation loss evolved to check whether our training is going in the right direction:

# In[ ]:


plt.figure(figsize=(20,12))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc='upper right', prop={'size': 24});


# The validation loss has been decreasing at a similar pace as the training loss, indicating that our model has been learning effectively over the last 30 epochs. We could try to train longer and see if the validation loss can be decreased further. Once the validation loss stops decreasing for a couple of epochs in a row, that's when we will want to stop training. Our final weights will then be the weights of the epoch that had the lowest validation loss.

# ### 5. Make predictions
# 
# Now let's make some predictions on the validation dataset with the trained model. For convenience we'll use the validation generator which we've already set up above. Feel free to change the batch size.
# 
# You can set the `shuffle` option to `False` if you would like to check the model's progress on the same image(s) over the course of the training.

# In[ ]:


# 1: Set the generator for the predictions.

predict_generator = val_dataset.generate(batch_size=1,
                                         shuffle=True,
                                         transformations=[],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'processed_labels',
                                                  'filenames'},
                                         keep_images_without_gt=False)


# In[ ]:


# 2: Generate samples

batch_images, batch_labels, batch_filenames = next(predict_generator)




i = 0 # Which batch item to look at
print("Image:", batch_filenames[i])
print()
print("Ground truth boxes:\n")
print(batch_labels[i])


# In[ ]:


for x in range(6):
  # 2: Generate samples
  print(x)
  batch_images, batch_labels, batch_filenames = next(predict_generator)
  i = 0 # Which batch item to look at
  print("Image:", batch_filenames[i])
  print()
  print("Ground truth boxes:\n")
  print(batch_labels[i])
  y_pred = model.predict(batch_images)
  y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.5,
                                   iou_threshold=0.45,
                                   top_k=200,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width)


# In[ ]:


for x in range(6):
  # 2: Generate samples
  batch_images, batch_labels, batch_filenames = next(predict_generator)
  i = 0 # Which batch item to look at
  print("Image:", batch_filenames[i])
  print()
  print("Ground truth boxes:\n")
  print(batch_labels[i])
  y_pred = model.predict(batch_images)
  # 4: Decode the raw prediction `y_pred`

  y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.5,
                                   iou_threshold=0.45,
                                   top_k=200,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width)

  np.set_printoptions(precision=2, suppress=True, linewidth=90)
  print("Predicted boxes:\n")
  print('   class   conf xmin   ymin   xmax   ymax')
  print(y_pred_decoded[i])


# In[ ]:


# 3: Make a prediction

y_pred = model.predict(batch_images)


# Now let's decode the raw predictions in `y_pred`.
# 
# Had we created the model in 'inference' or 'inference_fast' mode, then the model's final layer would be a `DecodeDetections` layer and `y_pred` would already contain the decoded predictions, but since we created the model in 'training' mode, the model outputs raw predictions that still need to be decoded and filtered. This is what the `decode_detections()` function is for. It does exactly what the `DecodeDetections` layer would do, but using Numpy instead of TensorFlow (i.e. on the CPU instead of the GPU).
# 
# `decode_detections()` with default argument values follows the procedure of the original SSD implementation: First, a very low confidence threshold of 0.01 is applied to filter out the majority of the predicted boxes, then greedy non-maximum suppression is performed per class with an intersection-over-union threshold of 0.45, and out of what is left after that, the top 200 highest confidence boxes are returned. Those settings are for precision-recall scoring purposes though. In order to get some usable final predictions, we'll set the confidence threshold much higher, e.g. to 0.5, since we're only interested in the very confident predictions.

# In[ ]:


# 4: Decode the raw prediction `y_pred`

y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.5,
                                   iou_threshold=0.45,
                                   top_k=200,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width)

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_decoded[i])


# Finally, let's draw the predicted boxes onto the image. Each predicted box says its confidence next to the category name. The ground truth boxes are also drawn onto the image in green for comparison.

# In[ ]:


# 5: Draw the predicted boxes onto the image

plt.figure(figsize=(20,12))
plt.imshow(batch_images[i])

current_axis = plt.gca()

colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist() # Set the colors for the bounding boxes
classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist', 'light'] # Just so we can print class names onto the image instead of IDs

# Draw the ground truth boxes in green (omit the label for more clarity)
for box in batch_labels[i]:
    xmin = box[1]
    ymin = box[2]
    xmax = box[3]
    ymax = box[4]
    label = '{}'.format(classes[int(box[0])])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))  
    #current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

# Draw the predicted boxes in blue
for box in y_pred_decoded[i]:
    xmin = box[-4]
    ymin = box[-3]
    xmax = box[-2]
    ymax = box[-1]
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})


# In[ ]:


for x in range(500):
  # 2: Generate samples
  batch_images, batch_labels, batch_filenames = next(predict_generator)
  i = 0 # Which batch item to look at
  print("Image:", batch_filenames[i])
  print()
  print("Ground truth boxes:\n")
  print(batch_labels[i])
  y_pred = model.predict(batch_images)
  # 4: Decode the raw prediction `y_pred`

  y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.5,
                                   iou_threshold=0.45,
                                   top_k=200,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width)

  np.set_printoptions(precision=2, suppress=True, linewidth=90)
  print("Predicted boxes:\n")
  print('   class   conf xmin   ymin   xmax   ymax')
  print(y_pred_decoded[i])
  # 5: Draw the predicted boxes onto the image

  plt.figure(figsize=(20,12))
  plt.imshow(batch_images[i])

  current_axis = plt.gca()

  colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist() # Set the colors for the bounding boxes
  classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist', 'light'] # Just so we can print class names onto the image instead of IDs

  # Draw the ground truth boxes in green (omit the label for more clarity)
  for box in batch_labels[i]:
      xmin = box[1]
      ymin = box[2]
      xmax = box[3]
      ymax = box[4]
      label = '{}'.format(classes[int(box[0])])
      current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))  
      #current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

  # Draw the predicted boxes in blue
  for box in y_pred_decoded[i]:
      xmin = box[-4]
      ymin = box[-3]
      xmax = box[-2]
      ymax = box[-1]
      color = colors[int(box[0])]
      label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
      current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
      current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})


# In[ ]:




