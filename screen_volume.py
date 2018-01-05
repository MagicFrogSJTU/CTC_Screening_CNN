
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
import ctc_convnet
import os
import SimpleITK
from data_input import SCREEN_VOLUME_SIZE
import polyp_def
import time
from scipy.ndimage import label
from scipy.ndimage import find_objects
import scipy.ndimage



def main(checkpoint_dir, record_dir, which, inference, Parameters):
    test_dir = record_dir+which
    with open(test_dir, 'r') as f:
        lines = f.readlines()
    rawCTlist = []
    for line in lines:
        line = line[:-1]
        for file in os.listdir(line):
            if file == 'oriInterpolatedCTData.raw' or file == 'InterpolatedCTData.raw':
                rawCTlist.append(os.path.join(line, file))
                break

    with tf.Graph().as_default():
        vol = tf.placeholder(tf.float32, shape=(1, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, 1))
        logits = inference(vol, False)

        variable_averages = tf.train.ExponentialMovingAverage(Parameters.MOVING_AVERAGE)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
                return

            for rawCTdir in rawCTlist:
                time_b = time.time()
                Volume = polyp_def.Volume_Data()
                basedir = rawCTdir[:rawCTdir.rfind('/')]
                print(basedir)
                Volume.Set_Directory(basedir)
                Volume.Load_Volume_Data()
                Volume.load_colon_mask()
                Volume.colon_mask_dilation()
                Volume.colon_mask = 0
                rawCTdata = Volume.rawdata
                colon_mask = Volume.dilated_colon_mask
                input_shape = Volume.shape

                score_map = np.zeros(input_shape)

                screen_step = 30
                x0 = 0
                x1 = x0 + SCREEN_VOLUME_SIZE
                while x1 <= input_shape[0]:
                    y0 = 0
                    y1 = y0 + SCREEN_VOLUME_SIZE
                    while y1 <= input_shape[1] :
                        z0 = 0
                        z1 = z0 + SCREEN_VOLUME_SIZE
                        while z1 <= input_shape[2]:
                            crop_vol = rawCTdata[x0:x1, y0:y1, z0:z1].astype(np.float32)
                            crop_colon_mask = colon_mask[x0:x1, y0:y1, z0:z1]
                            if np.sum(crop_colon_mask) < 10:
                                z0 += screen_step
                                z1 = z0 + SCREEN_VOLUME_SIZE
                                continue

                            resized_test_vol = np.reshape(crop_vol,
                                              [1, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, 1])

                            [result] = sess.run([logits], feed_dict={vol: resized_test_vol})
                            result = np.reshape(result, [SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE])
                            cenCrop = int((48 - 36) / 2)
                            score_map[x0 + cenCrop:x1 - cenCrop, y0 + cenCrop:y1 - cenCrop,
                            z0 + cenCrop:z1 - cenCrop] = np.maximum(
                                score_map[x0 + cenCrop:x1 - cenCrop, y0 + cenCrop:y1 - cenCrop,
                                z0 + cenCrop:z1 - cenCrop],
                                result[cenCrop:-cenCrop, cenCrop:-cenCrop, cenCrop:-cenCrop])

                            z0 += screen_step
                            z1 = z0 + SCREEN_VOLUME_SIZE
                        y0 += screen_step
                        y1 = y0 + SCREEN_VOLUME_SIZE
                    x0 += screen_step
                    #print(x0)
                    x1 = x0 + SCREEN_VOLUME_SIZE

                result = score_map * colon_mask

                dir = os.path.join(basedir, "score_map.nii.gz")
                img = SimpleITK.GetImageFromArray(result)
                SimpleITK.WriteImage(img, dir)

                print("time consumed %d" %(time.time()-time_b))
                #break




def Test(threshold, record_dir, ifwrite=False):
    test_dir =record_dir + "testVolumeRecord.txt"

    with open(test_dir, 'r') as f:
        lines = f.readlines()
    rawCTlist = []
    for line in lines:
        line = line[:-1]
        for file in os.listdir(line):
            if file == 'oriInterpolatedCTData.raw' or file == 'InterpolatedCTData.raw':
                rawCTlist.append(os.path.join(line, file))
                break
    print("size:", len(rawCTlist))

    num_correct_candidates = 0
    num_false_candidates = 0
    for rawCTdir in rawCTlist:
        time_b = time.time()
        Volume = polyp_def.Volume_Data()
        basedir = rawCTdir[:rawCTdir.rfind('/')]
        print(basedir)
        Volume.Set_Directory(basedir)
        if not Volume.load_polyp_mask():
            print("Wrong!!!!!!")
            raise EOFError
            continue
        score_map = Volume.load_score_map('score_map.nii.gz')
        print(score_map.shape)
        print(Volume.polyp_mask.shape)
        correct, polyp_num, false, output = score_map_processing(score_map, Volume.polyp_mask, threshold)
        if correct < polyp_num:
            print("not found!", rawCTdir, polyp_num, correct)
        num_correct_candidates += correct
        num_false_candidates += false
        if ifwrite:
            output_image = SimpleITK.GetImageFromArray(output)
            SimpleITK.WriteImage(output_image, os.path.join(basedir, "segmentation.nii.gz"))
        print("Time consumed: ", time.time()-time_b)


    print("Correct candidates totally: ", num_correct_candidates)
    print("False candidates totally: ", num_false_candidates)

def score_map_processing(score_map, polyp_mask, threshold = 0.99):
    label_thresholded = (score_map>threshold).astype(np.uint8)
    labels_vol, labels_num = label(label_thresholded, scipy.ndimage.generate_binary_structure(3,2))
    #print("number of candidates is: ", labels_num)
    objects = find_objects(labels_vol)             # Return slice. slice(begin,end,step)
    num_correct_candidates = 0
    num_false_candidates = 0

    #polyp_labels, polyp_num = label(polyp_mask, scipy.ndimage.generate_binary_structure(3,2))
    polyp_num = np.max(polyp_mask)
    polyp_found = np.linspace(1, polyp_num, polyp_num, dtype=np.int)
    #print(polyp_found)
    #print("Number of existing polyps is: ", polyp_num)

    for i in range(1,labels_num+1):
        object = objects[i-1]
        target_vol = (labels_vol[object])==i
        #print("No.", i)
        #size = np.sum(target_whole_vol[object])
        #print("size:",size)

        #center = scipy.ndimage.measurements.center_of_mass(label_thresholded, labels_vol, i)
        #center = np.array(center).astype(np.int)
        #print("center: ", center)
        #mean = scipy.ndimage.mean(score_map, labels_vol, i)
        #print("mean: ", mean)
        overlap = target_vol * polyp_mask[object]
        if np.sum(overlap) > 0:
            #print("It's a polyp!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            for j in range(1, polyp_num+1):
                if j in overlap:
                    if polyp_found[j-1] == j:
                        num_correct_candidates += 1
                        polyp_found[j-1] = 0

        else:
            num_false_candidates += 1
            #print("Not a polyp!!!!!!!!!!!!!!!!!!!!!!!")


    return num_correct_candidates, polyp_num, num_false_candidates, labels_vol

