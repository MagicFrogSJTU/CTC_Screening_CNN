# Created by Chen Yizhi
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import numpy as np
import tensorflow as tf
import os
import SimpleITK
import time
from scipy.ndimage import label
from scipy.ndimage import find_objects
import scipy.ndimage
import utils
from dataStructure import Volume_Data
SCREEN_CROP_LEN = 72 + 20

TestSize = 160

def screen_cnn(inference, db, cfg):
    with tf.Graph().as_default():
        vol = tf.placeholder(tf.float32, shape=(1, TestSize, TestSize, TestSize, 1))
        logits = inference(vol, False)

        variable_averages = tf.train.ExponentialMovingAverage(cfg.MOVING_AVERAGE)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            ckpt = tf.train.get_checkpoint_state(cfg.get_current_checkpoint_dir())
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
                return

            df = db.df.query('fold=="test"')
            # TODO: Combine those repeated.

            multi_i = -1
            for volume_uid, n_group in df.groupby('volume uid'):
                row = n_group.iloc[0]

                multi_i += 1
                if multi_i%cfg.multi_flag[0] != cfg.multi_flag[1]: # For multi processing
                    continue
                time_b = time.time()

                volume_data = Volume_Data()
                volume_data.Set_Directory(row['volume path'])
                volume_data.load_volume_data()
                volume_data.load_colon_dilation()
                rawCTdata = volume_data.CT_data
                colon_mask = volume_data.dilated_colon_mask
                input_shape = volume_data.shape

                rawCTdata = (rawCTdata+999)/2000.0 # !!Normalizing

                score_map = np.zeros(input_shape)

                screen_step = TestSize - 32
                x0 = 0
                while x0 <= input_shape[0]:
                    y0 = 0
                    print(x0)
                    while y0 <= input_shape[1] :
                        z0 = 0
                        while z0 <= input_shape[2]:

                            l_index = np.array([x0,y0,z0])
                            crop_vol = utils.crop(rawCTdata, l_index, TestSize, fill_value=-1)
                            crop_colon_mask = utils.crop(colon_mask, l_index, TestSize, fill_value=0)

                            if np.sum(crop_colon_mask) == 0:
                                z0 += screen_step
                                continue

                            resized_test_vol = np.reshape(crop_vol,
                                              [1, TestSize, TestSize, TestSize, 1])
                            [result] = sess.run([logits], feed_dict={vol: resized_test_vol})
                            result = np.squeeze(result)

                            cenCrop = 12
                            utils.revert_crop_MAXIMUM(score_map, l_index+cenCrop, TestSize-2*cenCrop, result)

                            z0 += screen_step
                        y0 += screen_step
                    x0 += screen_step

                result = score_map * colon_mask

                fold = os.path.join(cfg.screening_output_url,
                                    row['patient uid'],
                                    volume_uid,
                                    cfg.result_file_fold,)
                print(fold)
                if not os.path.exists(fold):
                    os.makedirs(fold)
                else:
                    shutil.rmtree(fold)
                    os.makedirs(fold)
                directory = os.path.join(fold, "score_map.nii.gz")
                img = SimpleITK.GetImageFromArray(result)
                SimpleITK.WriteImage(img, directory)

                # copy image and mask data, if not exist
                path = os.path.join(cfg.screening_output_url, row['patient uid'], volume_uid)
                if not os.path.exists(os.path.join(path, "CT_data.nii.gz")):
                    directory = os.path.join(path, "CT_data.nii.gz")
                    img = SimpleITK.GetImageFromArray(volume_data.CT_data)
                    SimpleITK.WriteImage(img, directory)

                    volume_data.load_polyp_mask()
                    directory = os.path.join(path, "polyp_mask.nii.gz")
                    img = SimpleITK.GetImageFromArray(volume_data.polyp_mask)
                    SimpleITK.WriteImage(img, directory)
                volume_data.clear_volume_data()
                print("time consumed %d" %(time.time()-time_b))
                #break




def analysis_of_screen(db, cfg, seed_threshold, grow_threshold=0.9, ):
    num_correct_candidates = 0
    num_false_candidates = 0
    num_gold_candidates = 0
    df = db.df.query('fold=="test"')
    multi_i = -1
    for volume_uid, n_group in df.groupby('volume uid'):
        row = n_group.iloc[0]

        multi_i += 1
        if multi_i%cfg.multi_flag[0] != cfg.multi_flag[1]:
            continue
        time_b = time.time()

        volume = Volume_Data()
        test_volume_path = os.path.join(cfg.screening_output_url,
                            row['patient uid'],
                            volume_uid,
                            cfg.result_file_fold,)
        volume.Set_Directory(test_volume_path)
        if not volume.load_polyp_mask():
            raise IOError
        volume.load_score_map(cfg.result_file_fold)
        volume.load_volume_data()

        segmentation(volume, seed_threshold, grow_threshold, cfg.result_file_fold)
        num_gold, num_correct , num_false = confirm(volume, cfg.result_file_fold)

        if num_correct< num_gold:
            print("not found!", volume.base_dir, num_gold, num_correct)
        num_correct_candidates += num_correct
        num_false_candidates += num_false
        num_gold_candidates += num_gold

        volume.clear_volume_data()
        print("Time consumed: ", time.time()-time_b)

    print("Correct candidates totally: ", num_correct_candidates)
    print("False candidates totally: ", num_false_candidates)

def segmentation(volume, seed_threshold, grow_threshold, result_file_fold=''):
    # Segmentation
    seed_area = (volume.score_map>seed_threshold).astype(np.uint8)
    region_grow_area = (volume.score_map>grow_threshold).astype(np.uint8)

    labels_vol, labels_num = label(region_grow_area, scipy.ndimage.generate_binary_structure(3,2))
    objects = find_objects(labels_vol)             # Return slice. slice(begin,end,step)

    num_candidates = 0
    segment_fold = os.path.join(volume.base_dir, result_file_fold, "segments")
    if not os.path.exists(segment_fold):
        os.mkdir(segment_fold)
    else:
        shutil.rmtree(segment_fold)
        os.mkdir(segment_fold)
    for i in range(labels_num):
        object = objects[i]
        select_label_whole = np.zeros(volume.CT_data.shape, dtype=np.uint8)
        select_label = labels_vol[object] == (i+1)
        select_label_whole[object] = select_label
        if np.sum(select_label*(seed_area[object])) == 0:
            continue

        center = scipy.ndimage.measurements.center_of_mass(region_grow_area, labels_vol, i+1)
        center = np.array(center).astype(np.int)
        left = center.copy()
        left -= int(SCREEN_CROP_LEN/2)
        #print(left)
        cropped_ct_data = utils.crop(volume.CT_data, left, SCREEN_CROP_LEN, -999)
        cropped_segment = utils.crop(volume.score_map,left, SCREEN_CROP_LEN, 0)
        cropped_select_label_whole = utils.crop(select_label_whole, left, SCREEN_CROP_LEN, 0)
        cropped_segment = cropped_segment * cropped_select_label_whole

        name = str(center[0])+"#"+str(center[1])+"#"+str(center[2])
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(cropped_ct_data),
                             os.path.join(segment_fold, name+"#ct"+".nii.gz"))
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(cropped_segment),
                             os.path.join(segment_fold, name+"#score"+".nii.gz"))
        with open(os.path.join(segment_fold, name+"#result"), 'w') as f:
            f.write("2")
        num_candidates += 1

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def confirm(volume, result_file_fold=''):
    polyp_num = np.max(volume.polyp_mask)
    for i in range(polyp_num):
        if np.sum(volume.polyp_mask == (i+1)) == 0:
            print("A label of mask is blank!", volume.base_dir, i)

    polyp_found = np.zeros((polyp_num))
    #print("Number of existing polyps is: ", polyp_num)
    num_false_candidates = 0

    segment_fold = os.path.join(volume.base_dir, result_file_fold, "segments")
    files = os.listdir(segment_fold)
    files = sorted(files)
    num_unit = 3
    assert len(files)%num_unit == 0

    for i in range(0, len(files), num_unit):
        #cropped_ct_data = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(
        #    os.path.join(segment_fold, files[i+1])))
        cropped_score_data = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(
            os.path.join(segment_fold, files[i+2])))
        name_str = files[i][:files[i].rfind('#')]
        temp = name_str.split('#')
        center = np.array(temp).astype(np.int)
        left = center.copy() - int(SCREEN_CROP_LEN/2)
        cropped_polyp_mask = utils.crop(volume.polyp_mask, left, SCREEN_CROP_LEN, 0)

        fopen = open(os.path.join(segment_fold, name_str+"#result"), 'w')
        overlap = cropped_polyp_mask * (cropped_score_data!=0)
        if np.sum(overlap) > 0:
            fopen.write("1")
            for j in range(polyp_num):
                if (j+1) in overlap:
                    if polyp_found[j] == 0:
                        polyp_found[j] = 1
        else:
            fopen.write("0")
            num_false_candidates += 1
        fopen.close()

    print("Number of false positives is:", num_false_candidates)
    return polyp_num, np.sum(polyp_found), num_false_candidates,

def produce_tf_samples(volume_manager, multi_flag, result_file_fold=''):

    num_correct_candidates = 0
    num_false_candidates = 0
    print("Total volumes:", len(volume_manager.volume_list))
    for index, volume in enumerate(volume_manager.volume_list):
        if index%multi_flag[0] != multi_flag[1]:
            continue
        time_b = time.time()
        #print(volume.base_dir)
        if not volume.load_polyp_mask():
            print("Wrong!!!!!!")
            raise EOFError
            continue

        volume.Load_Volume_Data()
        volume.load_polyp_mask()
        volume.load_colon_mask()

        num_correct, num_false = produce_tf_samples_unit(volume, result_file_fold)

        num_correct_candidates += num_correct
        num_false_candidates += num_false

        volume.clear_volume_data()
        print("Time consumed: ", time.time()-time_b)

    print("Correct candidates totally: ", num_correct_candidates)
    print("False candidates totally: ", num_false_candidates)



def produce_tf_samples_unit(volume, result_file_fold=''):
    segment_fold = os.path.join(volume.base_dir, result_file_fold, "segments")
    files = os.listdir(segment_fold)
    files = sorted(files)
    num_unit = 3
    assert len(files)%num_unit == 0

    writer1 = tf.python_io.TFRecordWriter(
        os.path.join(volume.base_dir, result_file_fold, 'true_screen_samples.tf'))
    writer2 = tf.python_io.TFRecordWriter(
        os.path.join(volume.base_dir, result_file_fold, 'false_screen_samples.tf'))
    num_true = 0
    num_false = 0
    for i in range(0, len(files), num_unit):
        cropped_ct_data = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(
            os.path.join(segment_fold, files[i+0])))
        cropped_score_data = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(
            os.path.join(segment_fold, files[i+2])))
        cropped_score_data = cropped_score_data!=0
        name_str = files[i][:files[i].rfind('#')]
        temp = name_str.split('#')
        center = np.array(temp).astype(np.int)
        left = center.copy() - int(SCREEN_CROP_LEN/2)
        cropped_polyp_mask = utils.crop(volume.polyp_mask, left, SCREEN_CROP_LEN, 0)
        center_label = volume.polyp_mask[center[0], center[1], center[2]]

        URL = volume.base_dir + "|" + name_str
        overlap = cropped_polyp_mask * (cropped_score_data!=0)
        if np.sum(overlap)>0:
            label_value = int(1)
            if center_label == 0:
                center_label = np.max(overlap)
                print("Inconsistence:", URL, center_label)
        else:
            label_value = int(0)
        example = tf.train.Example(features=tf.train.Features(feature={
            'volume':_bytes_feature(cropped_ct_data.astype(np.float32).tostring()),  # be float32!
            'score_map': _bytes_feature(cropped_score_data.astype(np.uint8).tostring()),
            'label': _int64_feature(label_value),
            'URL': _bytes_feature(URL),
            'polyp_label': _int64_feature(int(center_label)),
            'mask':_bytes_feature(cropped_polyp_mask.astype(np.uint8).tostring()), }))
        if np.sum(overlap)>0:
            writer1.write(example.SerializeToString())
            num_true += 1
        else:
            writer2.write(example.SerializeToString())
            num_false += 1

    writer1.close()
    writer2.close()
    return num_true, num_false
