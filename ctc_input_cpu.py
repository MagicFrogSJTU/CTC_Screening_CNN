''' Created by Yizhi Chen. 20171007'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from data_input import CUT_RAW_VOLUME_SIZE
from data_input import SCREEN_VOLUME_SIZE
import time
import polyp_def
import SimpleITK
from sample3D import transform3D

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


def read_data(filename_queue, Parameters):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'volume': tf.FixedLenFeature([], tf.string),
            'mask': tf.FixedLenFeature([], tf.string),
            'colon_mask': tf.FixedLenFeature([], tf.string)
        })
    volume = tf.decode_raw(features['volume'], tf.float32)
    volume.set_shape(POLYP_VOLUME_SIZE[0]*POLYP_VOLUME_SIZE[1]*POLYP_VOLUME_SIZE[2])
    volume = tf.reshape(volume, MASK_VOLUME_SIZE)


    mask = tf.decode_raw(features['mask'], tf.uint8)
    mask.set_shape(MASK_VOLUME_SIZE[0]*MASK_VOLUME_SIZE[1]*MASK_VOLUME_SIZE[2])
    mask = tf.reshape(mask, MASK_VOLUME_SIZE)

    colon_mask = tf.decode_raw(features['colon_mask'], tf.uint8)
    colon_mask.set_shape(POLYP_VOLUME_SIZE[0]*POLYP_VOLUME_SIZE[1]*POLYP_VOLUME_SIZE[2])
    colon_mask = tf.reshape(colon_mask, MASK_VOLUME_SIZE)

    mask = tf.cast(mask, tf.float32)

    croped_vol, croped_mask = tf.py_func(samples_augment, [volume, mask, colon_mask, Parameters.TRUE_SAMPLE_RATIO],
                                         [volume.dtype, mask.dtype], stateful=False)
    croped_vol.set_shape(MASK_SCREEN_SIZE)
    croped_mask.set_shape((SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, 1))
    croped_vol = tf.reshape(croped_vol, POLYP_SCREEN_SIZE)

    return croped_vol, croped_mask


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
        fopen = open(record_file_dir,"r")
        lines = fopen.readlines()
        file_names = []
        for line in lines:
            words = line[:-1].split()
            file_names.append(os.path.join(database_dir, words[0], words[1],
                              polyp_def.Polyp_data.augment_zoom_tf_name_prefix+".tf"))
        filename_queue = tf.train.string_input_producer(file_names)

        # Read examples from files in the filename queue.
        threads = 16
        example_list = [read_data(filename_queue, Parameters) for _ in range(threads)]

        return tf.train.shuffle_batch_join(
            example_list,
            batch_size=batch_size,
            capacity= NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN + 100 * batch_size,
            min_after_dequeue= NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)

    else:
        fopen = open(record_file_dir, "r")
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

def samples_augment(vol, mask, colon_mask, TRUE_SAMPLE_RATIO):
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
    afa = np.random.rand()*2*np.pi
    beta = np.random.rand()*2*np.pi
    croped_vol = transform3D(vol, afa=afa, beta=beta, output_size=MASK_SCREEN_SIZE,
                            origin=sample_center, defaultPixelValue=-999, interp_style='linear')
    croped_mask = transform3D(mask, afa=afa, beta=beta, output_size=MASK_SCREEN_SIZE,
                            origin=sample_center, defaultPixelValue=0, zoom_ratio=1, interp_style='linear')
    croped_mask = (croped_mask>0.25).astype(np.float32)

    # Flip in 3 axis randomly.
    for i in range(3):
        if np.random.rand() < 0.5:
            croped_vol = np.flip(croped_vol, axis=i)
            croped_mask = np.flip(croped_mask, axis=i)
    #if np.random.rand()<1:
    #    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(croped_vol), "/mnt/disk6/Yizhi/testVol.nii.gz")
    #    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(croped_mask.astype(np.uint8)), "/mnt/disk6/Yizhi/testMask.nii.gz")
    #    raise EOFError

    croped_mask = np.reshape(croped_mask,(SCREEN_VOLUME_SIZE,SCREEN_VOLUME_SIZE,SCREEN_VOLUME_SIZE,1))
    return croped_vol, croped_mask

