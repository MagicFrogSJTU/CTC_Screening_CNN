# Created by Chen Yizhi. By 2017.??.??

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def layer(input, name, channels, shape, strides=(1,1,1), ifBN=False, ifBNTrain=True, ifBNScale=False, ifPool=False, ifRelu=False):
    with tf.variable_scope(name) as scope:
        conv = tf.layers.conv3d(input, channels, shape, strides, padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32),
                                bias_initializer=tf.constant_initializer(0.001),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.005),)

        if ifBN:
            conv = tf.layers.batch_normalization(conv, training=ifBNTrain, scale=ifBNScale)
        if ifRelu:
            conv = tf.nn.relu(conv)
        if ifPool:
            conv = tf.layers.max_pooling3d(conv, [2,2,2], [2,2,2])
        return conv

def deconv_layer(input, name, channels, shape, strides=(1,1,1), ifBN=False, ifBNTrain=True, ifBNScale=False, ifRelu=True):
    with tf.variable_scope(name) as scope:
        conv = tf.layers.conv3d_transpose(input, channels, shape, strides, padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32),
                                bias_initializer=tf.constant_initializer(0.001),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.005))
        if ifRelu:
            conv = tf.nn.leaky_relu(conv)
        if ifBN:
            conv = tf.layers.batch_normalization(conv, training=ifBNTrain, scale=ifBNScale)
        if ifRelu:
            conv = tf.nn.relu(conv)
        return conv

def inference(volumes, iftrain):
    conv = volumes
    # 1
    with tf.name_scope('down1'):
        conv1 = layer(conv, 'conv1.0', 32, [3, 3, 3], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=True)
        conv1 = layer(conv1, 'conv1.1', 32, [3, 3, 3], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=False)
        conv = conv1[:,:,:,:,0:1] + conv
        conv = tf.concat([conv, conv1[:,:,:,:,1:32]], 4)
        conv = tf.nn.leaky_relu(conv)

        conv1 = layer(conv, 'conv1.2', 32, [3, 3, 3], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=True)
        conv1 = layer(conv1, 'conv1.3', 32, [3, 3, 3], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=False)
        conv = conv1 + conv
        conv_left1 = tf.nn.leaky_relu(conv)

        conv = layer(conv_left1, 'down1', 32, [2,2,2], [2,2,2], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=True)

    # 2
    with tf.name_scope('down2'):
        conv1 = layer(conv, 'conv2.0', 32, [3, 3, 3], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=True)
        conv1 = layer(conv1, 'conv2.1', 32, [3, 3, 3], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=False)
        conv = conv1 + conv
        conv = tf.nn.leaky_relu(conv)

        conv1 = layer(conv, 'conv2.2', 32, [3, 3, 3], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=True)
        conv1 = layer(conv1, 'conv2.3', 32, [3, 3, 3], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=False)
        conv = conv1 + conv
        conv_left2 = tf.nn.leaky_relu(conv)

        conv = layer(conv_left2, 'down2', 32, [2,2,2], [2,2,2], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=True)

    # 3
    with tf.name_scope('bottom'):
        conv1 = layer(conv, 'conv3.0', 32, [3, 3, 3], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=True)
        conv1 = layer(conv1, 'conv3.1', 32, [3, 3, 3], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=False)
        conv = conv1 + conv
        conv = tf.nn.leaky_relu(conv)

        # 3_
        conv1 = layer(conv, 'conv3.0_', 32, [3, 3, 3], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=True)
        conv1 = layer(conv1, 'conv3.1_', 32, [3, 3, 3], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=False)
        conv = conv1 + conv
        conv = tf.nn.leaky_relu(conv)
        conv = deconv_layer(conv, 'up2_', 32, [2,2,2], [2,2,2], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=True)

    # 2_
    with tf.name_scope('up2'):
        conv1 = tf.concat([conv, conv_left2], 4, name='concat2_')
        conv1 = layer(conv1, 'conv2.0_', 32, [3, 3, 3], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=True)
        conv1 = layer(conv1, 'conv2.1_', 32, [3, 3, 3], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=False)
        conv = conv1 + conv
        conv = tf.nn.leaky_relu(conv)

        conv1 = layer(conv, 'conv2.2_', 32, [3, 3, 3], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=True)
        conv1 = layer(conv1, 'conv2.3_', 32, [3, 3, 3], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=False)
        conv = conv1 + conv
        conv = tf.nn.leaky_relu(conv)


        conv = deconv_layer(conv, 'up1_', 16, [2,2,2], [2,2,2], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=True)

    # 1_
    with tf.name_scope('up1'):
        conv1 = tf.concat([conv, conv_left1], 4, name='concat1_')
        conv1 = layer(conv1, 'conv1.0_', 16, [3, 3, 3], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=True)
        conv1 = layer(conv1, 'conv1.1_', 16, [3, 3, 3], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=False)
        conv = conv1 + conv
        conv = tf.nn.leaky_relu(conv)

        conv1 = layer(conv, 'conv1.2_', 16, [3, 3, 3], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=True)
        conv1 = layer(conv1, 'conv1.3_', 16, [3, 3, 3], ifBN=True, ifBNTrain=iftrain, ifBNScale=True, ifRelu=False)
        conv = conv1 + conv
        conv = tf.nn.leaky_relu(conv)

    # Final
    with tf.name_scope('final'):
        conv = layer(conv, 'conv_final', 1, [1,1,1], ifRelu=False)
        conv = tf.sigmoid(conv)

    return conv










