''' Created by Yizhi Chen. 20171007'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import numpy as np
import os
from Configuration import CUT_RAW_VOLUME_SIZE
from Configuration import SCREEN_VOLUME_SIZE
import SimpleITK
import tensorflow.contrib.image as tfImage

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 200
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2

RAW_VOLUME_SIZE = [CUT_RAW_VOLUME_SIZE, CUT_RAW_VOLUME_SIZE, CUT_RAW_VOLUME_SIZE]
NETWORK_VOLUME_SIZE = [SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE]

MAX_RATIO = 3
sample_len = 22
print("The length of sampling translation is: %f" %(sample_len))
SAMPLE_POSITIVE_LOWER = int(   (CUT_RAW_VOLUME_SIZE - sample_len) / 2   )
SAMPLE_POSITIVE_HIGHER = int(  (CUT_RAW_VOLUME_SIZE + sample_len) / 2   )
SAMPLE_NEGATIVE_LOWER = int(   (CUT_RAW_VOLUME_SIZE - SCREEN_VOLUME_SIZE) / 2   )
SAMPLE_NEGATIVE_HIGHER = int(  (CUT_RAW_VOLUME_SIZE + SCREEN_VOLUME_SIZE) / 2   )

def data_augment(input, output_size, zoom_ratio, translation, afa, beta, if_flip):
    '''Scaling, Rotating, Flipping via Tensorflow api in 3d space.
    Args:
        #TODO
    '''
    outputsize3 = [CUT_RAW_VOLUME_SIZE, CUT_RAW_VOLUME_SIZE, CUT_RAW_VOLUME_SIZE]
    newsize = tf.cast((outputsize3)*zoom_ratio, tf.int32)
    #newsize = tf.concat([newlen, newlen, newlen], 0)
    rotate_input_len = int((3*0.5)*output_size)
    with tf.name_scope('resize'), tf.device('/gpu:0'):
        image_size = newsize[:2]
        first_output = tf.image.resize_images(input, image_size)

        transposed = tf.transpose(first_output, perm=[0,2,1])
        transposed = tf.image.resize_images(transposed, image_size)
        zoomed = tf.transpose(transposed, perm=[0,2,1])

    # Make sure the Slicing is done inside the input zone.
    trans_enable = tf.cast(tf.cast(newsize,tf.float32)/2- rotate_input_len/2, tf.int32)
    max_trans = tf.reduce_max(tf.abs(trans_enable))
    max_trans = tf.cast(max_trans, tf.float32)
    translation = tf.cast(tf.clip_by_value(translation, -max_trans, max_trans), tf.int32)
    crop_left = translation + trans_enable
    cropped_input = tf.slice(zoomed,
                             begin=crop_left,
                             size=[rotate_input_len, rotate_input_len, rotate_input_len])

    with tf.name_scope('rotate'), tf.device('/gpu:0'):
        with tf.name_scope('afa'):
            # afa.
            rotated_cropped_input = tfImage.rotate(cropped_input, afa, interpolation='BILINEAR')
        # beta.
        transposed_input = tf.transpose(rotated_cropped_input, perm=[0,2,1])
        with tf.name_scope('beta'):
            rotated_cropped_input = tf.contrib.image.rotate(transposed_input, beta, interpolation='BILINEAR')
        rotated_cropped_input = tf.cond(if_flip[0] < 0.5, lambda:tf.image.flip_left_right(rotated_cropped_input),
                                        lambda: rotated_cropped_input)
        rotated_cropped_input = tf.transpose(rotated_cropped_input, perm=[0,2,1])

        crop_l = int((rotate_input_len-output_size)/2)
        image = tf.slice(rotated_cropped_input,
                           begin=[crop_l,crop_l,crop_l],
                           size=[output_size,output_size,output_size])

    with tf.name_scope('flip'), tf.device('/gpu:0'):
        image = tf.cond(if_flip[1] < 0.5, lambda: tf.image.flip_up_down(image),
                          lambda: image)
        image = tf.cond(if_flip[2] < 0.5, lambda: tf.image.flip_left_right(image),
                          lambda: image)
        return image

def only_rotate3D(input, sample_center, size, afa, beta, if_flip):
    # TODO
    raise NotImplementedError
    '''Rotate via tensorflow api in 3d space.
    Args:
        size: Must be smaller than the longest bent axis of input. Must be size%4=0
    '''
    crop_left = sample_center - 36  # tf.constant((size*1.5/2.0).astype(np.int32))
    cropped_input = tf.slice(input, begin=crop_left, size=[72, 72, 72])

    with tf.name_scope('rotate'), tf.device('/gpu:0'):
        #size = np.array(size)
        with tf.name_scope('afa'):
            # afa.
            rotated_cropped_input = tf.contrib.image.rotate(cropped_input, afa, interpolation='BILINEAR')
        # beta.
        transposed_input = tf.transpose(rotated_cropped_input, perm=[0,2,1])
        with tf.name_scope('beta'):
            rotated_cropped_input = tf.contrib.image.rotate(transposed_input, beta, interpolation='BILINEAR')
        rotated_cropped_input = tf.cond(if_flip[0] < 0.5, lambda:tf.image.flip_left_right(rotated_cropped_input),
                                        lambda: rotated_cropped_input)
        rotated_cropped_input = tf.transpose(rotated_cropped_input, perm=[0,2,1])

        #sample_left = tf.constant((size*0.25).astype(np.int32))
        rotated = tf.slice(rotated_cropped_input, begin=[12,12,12], size=[48,48,48])
        rotated = tf.cond(if_flip[1] < 0.5, lambda: tf.image.flip_up_down(rotated),
                                        lambda: rotated)
        rotated = tf.cond(if_flip[2] < 0.5, lambda: tf.image.flip_left_right(rotated),
                                        lambda: rotated)
        return rotated

def read_data(filename_queue, mode, cfg):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'volume': tf.FixedLenFeature([], tf.string),
            'polyp_mask': tf.FixedLenFeature([], tf.string),
            'colon_mask': tf.FixedLenFeature([], tf.string),
        })

    volume = tf.decode_raw(features['volume'], tf.int16)
    volume.set_shape(RAW_VOLUME_SIZE[0]*RAW_VOLUME_SIZE[1]*RAW_VOLUME_SIZE[2])
    volume = tf.reshape(volume, RAW_VOLUME_SIZE)
    volume = tf.cast(volume, tf.float32)
    volume = (volume+999)/2000.0 # Normalization.


    mask = tf.decode_raw(features['polyp_mask'], tf.uint8)
    mask.set_shape(RAW_VOLUME_SIZE[0]*RAW_VOLUME_SIZE[1]*RAW_VOLUME_SIZE[2])
    mask = tf.reshape(mask, RAW_VOLUME_SIZE)
    mask = tf.cast(mask, tf.float32)

    colon_mask = tf.decode_raw(features['colon_mask'], tf.uint8)
    colon_mask.set_shape(RAW_VOLUME_SIZE[0]*RAW_VOLUME_SIZE[1]*RAW_VOLUME_SIZE[2])
    colon_mask = tf.reshape(colon_mask, RAW_VOLUME_SIZE)

    if mode == 'train':
        global_step = tf.train.get_or_create_global_step()
        ratio = tf.cond(global_step<1000, lambda:tf.constant(1.0), lambda:tf.constant(0.5))
        mask_size = tf.reduce_sum(mask)
        scaling, translation= tf.py_func(sampling, [mask, mask_size, colon_mask, cfg.MAX_POLYP_SIZE,
                                         cfg.MIN_POLYP_SIZE, ratio], [tf.float32, tf.float32], stateful=False)
        translation.set_shape([3])
        scaling.set_shape([1])

        # rotate
        degrees = tf.random_uniform([2], minval=0, maxval=2*np.pi, dtype=tf.float32)
        afa = degrees[0]
        beta = degrees[1]

        if_flip = tf.random_uniform([3], minval=0, maxval=1)
        croped_vol = data_augment(volume, SCREEN_VOLUME_SIZE, scaling, translation, afa, beta, if_flip)
        croped_mask = data_augment(mask, SCREEN_VOLUME_SIZE, scaling, translation, afa, beta, if_flip)

        croped_vol.set_shape(NETWORK_VOLUME_SIZE)
        croped_vol = tf.expand_dims(croped_vol, 3)
        croped_mask.set_shape(NETWORK_VOLUME_SIZE)
        croped_mask = tf.expand_dims(croped_mask, 3)
        croped_mask = tf.cast(croped_mask > 0.5, tf.float32)

        # Debug
        if 0:
            croped_vol = tf.py_func(test, [croped_vol, croped_mask], croped_vol.dtype)
            croped_vol.set_shape(NETWORK_VOLUME_SIZE)
            croped_vol = tf.reshape(croped_vol, NETWORK_VOLUME_SIZE)
        return croped_vol, croped_mask
    elif mode == 'validation':
        ini_index = int((CUT_RAW_VOLUME_SIZE-SCREEN_VOLUME_SIZE)/2)
        croped_vol = tf.slice(volume, [ini_index, ini_index, ini_index], NETWORK_VOLUME_SIZE)
        croped_mask = tf.slice(mask, [ini_index, ini_index, ini_index], NETWORK_VOLUME_SIZE)
        croped_vol = tf.expand_dims(croped_vol, 3)
        croped_mask = tf.expand_dims(croped_mask, 3)
        return croped_vol, croped_mask

def test(vol, mask):
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(vol), "testVol.nii.gz")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(mask), "testMask.nii.gz")
    raise EOFError
    return vol


def inputs(mode, batch_size, cfg, database):
    '''Read input data num_epochs time'''
    if mode == 'train':
        indexs = database.df.query('fold=="train"')['polyp index'].values.tolist()
        lines = []
        for index in indexs:
            line = os.path.join(cfg.polypdata_fold_url, str(index), 'data.tf')
            lines.append(line)
        filename_queue = tf.train.string_input_producer(lines)

        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

        # read examples from files in the filename queue.
        threads = 4
        example_list = [read_data(filename_queue, mode, cfg) for _ in range(threads)]

        return tf.train.shuffle_batch_join(
            example_list,
            batch_size=batch_size,
            capacity=num_examples_per_epoch + 15 * batch_size,
            min_after_dequeue=num_examples_per_epoch)
    elif mode == 'validation':
        indexs = database.df.query('fold=="validation"')['polyp index'].values.tolist()
    elif mode == 'test':
        indexs = database.df.query('fold=="test"')['polyp index'].values.tolist()
    else:
        raise ValueError

    lines = []
    for index in indexs:
        line = os.path.join(cfg.polypdata_fold_url, str(index), 'data.tf')
        lines.append(line)
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    filename_queue = tf.train.string_input_producer(lines, num_epochs=1)
    return tf.train.batch(read_data(filename_queue, mode, cfg),
                          batch_size=1,
                          num_threads=2)


def sampling(mask, polyp_size, colon_mask, MAX_POLYP_SIZE, MIN_POLYP_SIZE, TRUE_SAMPLE_RATIO):
    '''
    '''
    # scaling ratio
    radius = (polyp_size*3.0/4/np.pi)**(1/3.0)
    max_ = np.max([MAX_POLYP_SIZE*0.6/radius, 1.2])
    max_ = np.min([2, max_])
    min_ = np.min([radius/(MIN_POLYP_SIZE*2), 0.8])
    min_ = np.max([min_, 0.6])
    scaling_ratio = np.random.random()*(max_-min_)+min_




    if np.random.rand() < TRUE_SAMPLE_RATIO:
        translation = np.random.randint(-sample_len, sample_len, (3))
    else:
        count = 0
        while (True):
            count += 1
            translation = np.random.randint(int(-CUT_RAW_VOLUME_SIZE/2 + SCREEN_VOLUME_SIZE*0.25),
                                           int(CUT_RAW_VOLUME_SIZE/2-SCREEN_VOLUME_SIZE*0.25),
                                           (3))
            begin1d = int(CUT_RAW_VOLUME_SIZE/2 - SCREEN_VOLUME_SIZE/2)
            index_left = np.array([begin1d, begin1d, begin1d]) + translation
            index_right = index_left + SCREEN_VOLUME_SIZE
            cut = mask[index_left[0]:index_right[0],
                         index_left[1]:index_right[1],
                         index_left[2]:index_right[2]]
            colon_mask_cut = colon_mask[index_left[0]:index_right[0],
                         index_left[1]:index_right[1],
                         index_left[2]:index_right[2]]
            if np.sum(cut) < 0.1 and np.sum(colon_mask_cut)>30000:
                break
            if count>50:
                print("Test. Sampling Loop overstack")
                translation = np.random.randint(-sample_len, sample_len, (3))
                break
    translation = translation.astype(np.float32)
    scaling_ratio = scaling_ratio.astype(np.float32)

    return scaling_ratio, translation
