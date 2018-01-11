
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
import data_input
import ctc_convnet
import os
import ctc_input
import SimpleITK
import time
from data_input import SCREEN_VOLUME_SIZE
from data_input import CUT_RAW_VOLUME_SIZE
from scipy.ndimage import label
from scipy.ndimage import find_objects
import scipy.ndimage
from sample3D import transform3D
from dataDirectory import DataDirectory

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def crop(volume, output_len, l_index, fill_value=0):
    result = np.ones((output_len, output_len, output_len), dtype=volume.dtype)*fill_value
    copy0 = np.array(l_index)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! quote!
    copy1 = np.array(copy0)+output_len
    paste0 = [0, 0, 0]
    paste1 = [output_len, output_len, output_len]

    for i in range(3):
        if copy0[i]<0:
            paste0[i] = -copy0[i]
            copy0[i] = 0
        if copy1[i]<0:
            raise IndexError
        if copy1[i]>volume.shape[i]:
            paste1[i] = output_len - (copy1[i]-volume.shape[i])
            copy1[i] = volume.shape[i]
        if copy0[i]>volume.shape[i]:
            raise IndexError
    result[paste0[0]:paste1[0],paste0[1]:paste1[1], paste0[2]:paste1[2]] = \
        volume[copy0[0]:copy1[0], copy0[1]:copy1[1], copy0[2]:copy1[2]]
    return result

def ScreenRatio(record_dir, ratio=0.3):
    dataDirectory = DataDirectory()
    checkpoint_dir = os.path.join(record_dir, dataDirectory.checkpoint_fold)
    polyp_manager = data_input.Polyp_Manager()
    polyp_manager.read_polyps_from_disk_cross_val(record_dir,'all')
    print("Polyps loading complete!")
    with tf.Graph().as_default():
        vol = tf.placeholder(tf.float32, shape=(1, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, 1))
        logits = ctc_convnet.inference(vol, False)

        variable_averages = tf.train.ExponentialMovingAverage(ctc_convnet.MOVING_AVERAGE)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.4
        config.gpu_options.allow_growth = True

        with tf.Session(config=config,) as sess:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
                return

            for polyp in polyp_manager.polyp_list:
                time_b = time.time()

                vol_input = polyp.polyp_crop_data.astype(np.float32)
                original_shape = vol_input.shape
                input_shape = (np.array(original_shape)/ratio).astype(np.int)
                vol_input = transform3D(vol_input,
                                        output_size=input_shape,
                                        zoom_ratio=ratio,
                                        defaultPixelValue=-999,
                                        )
                count_map = np.zeros(input_shape) + 0.0001
                score_map = np.zeros(input_shape)
                screen_step = 24
                x0 = 0
                x1 = x0 + SCREEN_VOLUME_SIZE
                while x1 <= input_shape[0]:
                    y0 = 0
                    y1 = y0 + SCREEN_VOLUME_SIZE
                    while y1 <= input_shape[1]:
                        z0 = 0
                        z1 = z0 + SCREEN_VOLUME_SIZE
                        while z1 <= input_shape[2]:
                            crop_vol = vol_input[x0:x1, y0:y1, z0:z1]
                            resized_test_vol = np.reshape(crop_vol,
                                              [1, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, 1])
                            [result] = sess.run([logits], feed_dict={vol: resized_test_vol})
                            result = np.reshape(result, [SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE])
                            cenCrop = int((SCREEN_VOLUME_SIZE - 24)/2)
                            score_map[x0+cenCrop:x1-cenCrop, y0+cenCrop:y1-cenCrop,
                                z0+cenCrop:z1-cenCrop] += result[cenCrop:-cenCrop,cenCrop:-cenCrop, cenCrop:-cenCrop]
                            count_map[x0 + cenCrop:x1 - cenCrop, y0 + cenCrop:y1 - cenCrop,
                                      z0 + cenCrop:z1 - cenCrop] += 1


                            z0 += screen_step
                            z1 = z0 + SCREEN_VOLUME_SIZE
                        y0 += screen_step
                        y1 = y0 + SCREEN_VOLUME_SIZE
                    x0 += screen_step
                    x1 = x0 + SCREEN_VOLUME_SIZE
                score_map = score_map/count_map
                score_map = transform3D(score_map,
                                        output_size=original_shape,
                                        zoom_ratio = 1.0/ratio,
                                        defaultPixelValue=-999,)
                output = (score_map>10000).astype(np.uint8)

                output = output * polyp.dilated_colon_mask_data

                dir = os.path.join(polyp.base_folder_dir, "prediction.nii.gz")
                print(dir)
                img = SimpleITK.GetImageFromArray(np.transpose(output,(2,1,0)))
                SimpleITK.WriteImage(img, dir)

                dir = os.path.join(polyp.base_folder_dir, "score_map_ratio_"+str(ratio)+".nii.gz")
                img = SimpleITK.GetImageFromArray(np.transpose(score_map,(2,1,0)))
                SimpleITK.WriteImage(img, dir)
                print(time.time()-time_b)

def main(checkpoint_dir, record_dir, whichone, inference, Parameters):
    polyp_manager = data_input.Polyp_Manager()
    polyp_manager.read_polyps_from_disk(record_dir, whichone)
    print("Polyps loading complete!")
    with tf.Graph().as_default():
        vol = tf.placeholder(tf.float32, shape=(1, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, 1))
        logits = inference(vol, False)

        variable_averages = tf.train.ExponentialMovingAverage(Parameters.MOVING_AVERAGE)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.4
        config.gpu_options.allow_growth = True

        with tf.Session(config=config,) as sess:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
                return

            for polyp in polyp_manager.polyp_list:
                time_b = time.time()
                # For test.
                #if polyp.INDEX != testIndex:
                #    continue;
                score_map = np.zeros((CUT_RAW_VOLUME_SIZE,CUT_RAW_VOLUME_SIZE,CUT_RAW_VOLUME_SIZE))
                vol_input = polyp.polyp_crop_data.astype(np.float32)
                screen_step = 16
                x0 = 0
                x1 = x0 + SCREEN_VOLUME_SIZE
                while x1 <= CUT_RAW_VOLUME_SIZE:
                    y0 = 0
                    y1 = y0 + SCREEN_VOLUME_SIZE
                    while y1 <= CUT_RAW_VOLUME_SIZE:
                        z0 = 0
                        z1 = z0 + SCREEN_VOLUME_SIZE
                        while z1 <= CUT_RAW_VOLUME_SIZE:
                            crop_vol = vol_input[x0:x1, y0:y1, z0:z1]
                            resized_test_vol = np.reshape(crop_vol,
                                              [1, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, 1])
                            [result] = sess.run([logits], feed_dict={vol: resized_test_vol})
                            result = np.reshape(result, [SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE])
                            cenCrop = int((48-24)/2)
                            score_map[x0+cenCrop:x1-cenCrop, y0+cenCrop:y1-cenCrop,
                                z0+cenCrop:z1-cenCrop] = np.maximum(score_map[x0+cenCrop:x1-cenCrop, y0+cenCrop:y1-cenCrop,
                                z0+cenCrop:z1-cenCrop], result[cenCrop:-cenCrop,cenCrop:-cenCrop, cenCrop:-cenCrop])

                            z0 += screen_step
                            z1 = z0 + SCREEN_VOLUME_SIZE
                        y0 += screen_step
                        y1 = y0 + SCREEN_VOLUME_SIZE
                    x0 += screen_step
                    #print(x0)
                    x1 = x0 + SCREEN_VOLUME_SIZE

                #aver_score_map = np.zeros((CUT_RAW_VOLUME_SIZE,CUT_RAW_VOLUME_SIZE,CUT_RAW_VOLUME_SIZE))
                #np.divide(score_map, count_map, out=aver_score_map, where=count_map!=0)
                #score_map = aver_score_map * polyp.dilated_colon_mask_data
                score_map = score_map * polyp.dilated_colon_mask_data

                dir = os.path.join(polyp.base_folder_dir, "score_map.nii.gz")
                print(dir)
                img = SimpleITK.GetImageFromArray(score_map)
                SimpleITK.WriteImage(img, dir)
                print(time.time()-time_b)

def Test(polyp_manager, threshold, ifwrite=False):
    num_correct_candidates = 0
    num_false_candidatas = 0
    num_polyps = []
    for polyp in polyp_manager.polyp_list:
        #if polyp.INDEX != 224:
            #continue
        #print("-------------------------------------------------------------INDEX:", polyp.INDEX, "----")
        score_map_dir = os.path.join(polyp.base_folder_dir, "score_map.nii.gz")
        score_map1 = SimpleITK.GetArrayFromImage(SimpleITK
                                                 .ReadImage(score_map_dir))
        #score_map1 = np.where(score_map1>500, score_map1, 0)

        #score_map_dir = os.path.join(polyp.base_folder_dir, "score_map_ratio_2.nii.gz")
        #score_map2 = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(score_map_dir))
        #score_map2 = np.where(score_map2>500, score_map2, 0)

        #afa = 1     # Linear Variable.
        #beta = 1
        #count = (score_map1!=0).astype(np.float32)*afa + (score_map2!=0).astype(np.float32)*beta + 0.00001
        #score_map = (score_map1*afa + score_map2*beta) / count
        #count = 0
        #score_map1 = 0
        #score_map2 = 0
        correct, false, polyp_num, output = score_map_processing(score_map1, polyp.mask, threshold)
        if ifwrite:
            output_image = SimpleITK.GetImageFromArray(output)
            SimpleITK.WriteImage(output_image, os.path.join(polyp.base_folder_dir, "segmentation.nii.gz"))
        num_correct_candidates += correct
        num_false_candidatas += false
        num_polyps.append(polyp_num)

    print("Correct candidates totally:", num_correct_candidates)
    print("False candidates totally:", num_false_candidatas)
    num_polyps.sort()
    #print(num_polyps)


def score_map_processing(score_map, polyp_mask, threshold, dilation_threshold=0.5):
    label_thresholded = (score_map>threshold).astype(np.uint8)
    labels_vol, labels_num = label(label_thresholded, scipy.ndimage.generate_binary_structure(3,2))
    #print("number of candidates is: ", labels_num)
    objects = find_objects(labels_vol)             # Return slice. slice(begin,end,step)
    num_correct_candidates = 0
    num_false_candidatas = 0

    polyp_labels, polyp_num = label(polyp_mask, scipy.ndimage.generate_binary_structure(3,2))
    #print("Number of existing polyps is: ", polyp_num)
    size = np.sum(polyp_labels[find_objects(polyp_labels)[0]]!=0)
    output = np.zeros(score_map.shape, dtype=np.uint8)
    for i in range(1,labels_num+1):
        object = objects[i-1]
        target_whole_vol = labels_vol==i

        target_whole_vol = scipy.ndimage.binary_dilation(target_whole_vol, scipy.ndimage.generate_binary_structure(3,3),
                                                         1000, score_map>dilation_threshold).astype(np.uint8)
        #print("No.", i)
        #size = np.sum(target_whole_vol[object])
        #print("size:",size)

        #center = scipy.ndimage.measurements.center_of_mass(label_thresholded, labels_vol, i)
        #center = np.array(center).astype(np.int)
        #print("center: ", center)
        #mean = scipy.ndimage.mean(score_map, labels_vol, i)
        #print("mean: ", mean)
        output = output + target_whole_vol*i
        overlap = target_whole_vol[object] * polyp_labels[object]
        if np.sum(overlap) > 0:
            for j in range(polyp_num):
                if (j+1) in overlap:
                    polyp_labels = np.where(polyp_labels==(j+1), 0, polyp_labels)
        else:
            num_false_candidatas += 1

    if polyp_labels[80,80,80] == 0:
        num_correct_candidates = 1
    else:
        pass
        #print("Not found!")
        #print("size:", size)

    return num_correct_candidates, num_false_candidatas, score_map[80, 80, 80], output

if __name__ == '__main__':
    dataDirectory = DataDirectory()
    train_dir = os.path.join(dataDirectory.get_current_model_dir(),
                             dataDirectory.checkpoint_fold)
    record_dir = dataDirectory.get_current_record_dir()

    polyp_manager = data_input.Polyp_Manager()
    polyp_manager.read_polyps_from_disk(record_dir, 'all')

    main(train_dir, record_dir)

    #ScreenRatio(checkpoint_dir)+
    Test(polyp_manager, 0.99)
    Test(polyp_manager, 0.9, ifwrite=False)
    Test(polyp_manager, 0.8)
    Test(polyp_manager, 0.6)
    Test(polyp_manager, 0.5)
    #Test(polyp_manager, 0.3)
