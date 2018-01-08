''' Created by Yizhi Chen. 20171007'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from data_input import CUT_RAW_VOLUME_SIZE
from data_input import SCREEN_VOLUME_SIZE
import polyp_def
import SimpleITK

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 200
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2
POLYP_VOLUME_SIZE = [CUT_RAW_VOLUME_SIZE, CUT_RAW_VOLUME_SIZE, CUT_RAW_VOLUME_SIZE, 1]
MASK_VOLUME_SIZE = [CUT_RAW_VOLUME_SIZE, CUT_RAW_VOLUME_SIZE, CUT_RAW_VOLUME_SIZE]
POLYP_SCREEN_SIZE = [SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, 1]
MASK_SCREEN_SIZE = [SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE]

MAX_RATIO = 3
sample_len = 22
print("The length of sampling translation is: %f" %(sample_len))
SAMPLE_POSITIVE_LOWER = int(   (CUT_RAW_VOLUME_SIZE - sample_len) / 2   )
SAMPLE_POSITIVE_HIGHER = int(  (CUT_RAW_VOLUME_SIZE + sample_len) / 2   )
SAMPLE_NEGATIVE_LOWER = int(   (CUT_RAW_VOLUME_SIZE - SCREEN_VOLUME_SIZE) / 2   )
SAMPLE_NEGATIVE_HIGHER = int(  (CUT_RAW_VOLUME_SIZE + SCREEN_VOLUME_SIZE) / 2   )

def rotate_3d(input, sample_center, size, afa, beta, if_flip):
    '''Rotate via tensorflow api in 3d space.
    Args:
        size: Must be smaller than the longest bent axis of input. Must be size%4=0
    '''
    crop_left = sample_center - 36  # tf.constant((size*1.5/2.0).astype(np.int32))
    cropped_input = tf.slice(input, begin=crop_left, size=[72, 72, 72])

    with tf.name_scope('rotate'), tf.device('/gpu:2'):
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

def read_data(filename_queue, Parameters):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'volume': tf.FixedLenFeature([], tf.string),
            'mask': tf.FixedLenFeature([], tf.string),
            'colon_mask': tf.FixedLenFeature([], tf.string),
        })

    volume = tf.decode_raw(features['volume'], tf.float32)
    volume.set_shape(MASK_VOLUME_SIZE[0]*MASK_VOLUME_SIZE[1]*MASK_VOLUME_SIZE[2])
    volume = tf.reshape(volume, MASK_VOLUME_SIZE)


    mask = tf.decode_raw(features['mask'], tf.uint8)
    mask.set_shape(MASK_VOLUME_SIZE[0]*MASK_VOLUME_SIZE[1]*MASK_VOLUME_SIZE[2])
    mask = tf.reshape(mask, MASK_VOLUME_SIZE)
    mask = tf.cast(mask, tf.float32)

    colon_mask = tf.decode_raw(features['colon_mask'], tf.uint8)
    colon_mask.set_shape(POLYP_VOLUME_SIZE[0]*POLYP_VOLUME_SIZE[1]*POLYP_VOLUME_SIZE[2])
    colon_mask = tf.reshape(colon_mask, MASK_VOLUME_SIZE)

    sample_center = tf.py_func(samples_augment, [mask, colon_mask, Parameters.TRUE_SAMPLE_RATIO], tf.int32, stateful=False)
    sample_center.set_shape([3])

    # rotate
    afa = tf.random_uniform([1], minval=0, maxval=2 * np.pi)
    beta = tf.random_uniform([1], minval=0, maxval=2 * np.pi)
    if_flip = tf.random_uniform([3], minval=0, maxval=1)
    croped_vol = rotate_3d(volume, sample_center, MASK_SCREEN_SIZE, afa, beta, if_flip)
    croped_mask = rotate_3d(mask, sample_center, MASK_SCREEN_SIZE, afa, beta, if_flip)
    #croped_vol.set_shape(MASK_SCREEN_SIZE)
    #
    croped_vol.set_shape(MASK_SCREEN_SIZE)
    croped_vol = tf.reshape(croped_vol, POLYP_SCREEN_SIZE)
    croped_mask.set_shape(MASK_SCREEN_SIZE)
    croped_mask = tf.reshape(croped_mask, POLYP_SCREEN_SIZE)
    croped_mask = tf.cast(croped_mask > 0.5, tf.float32)

    #croped_vol = tf.py_func(test, [croped_vol, croped_mask], croped_vol.dtype)
    #croped_vol.set_shape(MASK_SCREEN_SIZE)
    #croped_vol = tf.reshape(croped_vol, POLYP_SCREEN_SIZE)
    return croped_vol, croped_mask

def test(vol, mask):
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(vol), "testVol.nii.gz")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(mask), "testMask.nii.gz")
    raise EOFError
    return vol


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle=1):
    num_preprocess_threads = 12
    if shuffle:
        images, label_batch = tf.train.shuffle_batch_join(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 1 * batch_size)
    return images, label_batch


def inputs(train_or_eval, batch_size, record_file_dir, database_dir, Parameters):
    '''Read input data num_epochs time'''
    if train_or_eval:
        fopen = open(record_file_dir)
        lines = fopen.readlines()
        file_names = []
        for line in lines:
            words = line[:-1].split()
            file_names.append(os.path.join(database_dir, words[0], words[1],
                              polyp_def.Polyp_data.augment_zoom_tf_name_prefix+".tf"))
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        filename_queue = tf.train.string_input_producer(file_names)

        # Read examples from files in the filename queue.
        threads = 8
        example_list = [read_data(filename_queue, Parameters) for _ in range(threads)]

        return tf.train.shuffle_batch_join(
            example_list,
            batch_size=batch_size,
            capacity=num_examples_per_epoch + 15 * batch_size,
            min_after_dequeue=num_examples_per_epoch)

    else:
        fopen = open(record_file_dir)
        lines = fopen.readlines()
        file_names = []
        for line in lines:
            words = line[:-1].split()
            file_names.append(os.path.join(database_dir, words[0], words[1],
                                           polyp_def.Polyp_data.raw_tf_name))
        #print(file_names)
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        filename_queue = tf.train.string_input_producer(file_names, num_epochs=1)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'volume': tf.FixedLenFeature([], tf.string),
                'mask': tf.FixedLenFeature([], tf.string),
            })
        volume = tf.decode_raw(features['volume'], tf.float32)
        volume.set_shape(POLYP_VOLUME_SIZE[0] * POLYP_VOLUME_SIZE[1] * POLYP_VOLUME_SIZE[2])
        volume = tf.reshape(volume, MASK_VOLUME_SIZE)
        mask = tf.decode_raw(features['mask'], tf.uint8)
        mask.set_shape(MASK_VOLUME_SIZE[0] * MASK_VOLUME_SIZE[1] * MASK_VOLUME_SIZE[2])
        mask = tf.reshape(mask, MASK_VOLUME_SIZE)
        reshaped_volume = tf.cast(volume, tf.float32)
        reshaped_mask = tf.cast(mask, tf.float32)

        croped_vol, croped_mask = tf.py_func(samples_test, [reshaped_volume, reshaped_mask],
                                             [reshaped_volume.dtype, reshaped_volume.dtype])
        croped_vol.set_shape(MASK_SCREEN_SIZE)
        croped_mask.set_shape((SCREEN_VOLUME_SIZE,SCREEN_VOLUME_SIZE,SCREEN_VOLUME_SIZE,1))
        croped_vol = tf.reshape(croped_vol, POLYP_SCREEN_SIZE)
        return _generate_image_and_label_batch(croped_vol, croped_mask,
                                               num_examples_per_epoch, batch_size,
                                           shuffle=False)



def samples_test(vol, mask):
    index_left = np.zeros((3),dtype=np.int)
    for i in range(3):
        index_left[i] = int((CUT_RAW_VOLUME_SIZE-SCREEN_VOLUME_SIZE)/2)
    index_right = index_left + SCREEN_VOLUME_SIZE
    croped_vol = vol[index_left[0]:index_right[0],
                 index_left[1]:index_right[1],
                 index_left[2]:index_right[2]]
    croped_mask = mask[index_left[0]:index_right[0],
                  index_left[1]:index_right[1],
                  index_left[2]:index_right[2]]
    croped_mask = croped_mask.astype(np.float32)
    croped_mask = np.reshape(croped_mask, (SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, 1))
    return croped_vol, croped_mask

def samples_augment(mask, colon_mask, TRUE_SAMPLE_RATIO):
    '''Traditional data augmentation.
    '''

    if np.random.rand() < TRUE_SAMPLE_RATIO:
        sample_center = np.random.randint(SAMPLE_POSITIVE_LOWER, SAMPLE_POSITIVE_HIGHER, (3))
    else:
        count = 0
        while (True):
            count += 1
            index_left = np.random.randint(int(SCREEN_VOLUME_SIZE*0.25),
                                           CUT_RAW_VOLUME_SIZE-SCREEN_VOLUME_SIZE-1-int(SCREEN_VOLUME_SIZE*0.25),
                                           (3))
            index_right = index_left + SCREEN_VOLUME_SIZE
            cut = mask[index_left[0]:index_right[0],
                         index_left[1]:index_right[1],
                         index_left[2]:index_right[2]]
            colon_mask_cut = colon_mask[index_left[0]:index_right[0],
                         index_left[1]:index_right[1],
                         index_left[2]:index_right[2]]
            if np.sum(cut) < 0.5 and np.sum(colon_mask_cut)>10000:
                sample_center = index_left + int(SCREEN_VOLUME_SIZE/2)
                break
            if count>50:
                sample_center = np.random.randint(SAMPLE_POSITIVE_LOWER, SAMPLE_POSITIVE_HIGHER, (3))
                print("Times of counting excess!!")
                break

    return sample_center.astype(np.int32)
