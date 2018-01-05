
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
checkpoint_dir = '/mnt/disk6/Yizhi/cross_validation/6/Screen/Train'
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dataDirectory import DataDirectory

def main():
    dataDirectory = DataDirectory()
    record_dir = dataDirectory.cross_validation_dir()
    polyp_manager = data_input.Polyp_Manager()
    polyp_manager.read_polyps_from_disk(dataDirectory.cross_validation_dir(), [40,0])
    print("Polyps loading complete!")
    with tf.Graph().as_default():
        vol = tf.placeholder(tf.float32, shape=(1, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, 1))
        logits = ctc_convnet.inference(vol, False)

        saver = tf.train.Saver()
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

                l_index = int((CUT_RAW_VOLUME_SIZE-SCREEN_VOLUME_SIZE)/2)
                r_index = l_index + SCREEN_VOLUME_SIZE
                crop_vol = vol_input[l_index:r_index, l_index:r_index, l_index:r_index]
                resized_test_vol = np.reshape(crop_vol,
                            [1, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, 1])
                [result] = sess.run([logits], feed_dict={vol: resized_test_vol})
                result = np.reshape(result, [SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE, SCREEN_VOLUME_SIZE])

                score_map[l_index:r_index, l_index:r_index, l_index:r_index] += result


                dir = os.path.join(polyp.base_folder_dir, "prediction.nii.gz")
                print(dir)

                img = SimpleITK.GetImageFromArray(score_map)
                SimpleITK.WriteImage(img, dir)



                print(time.time()-time_b)


if __name__ == '__main__':
    main()
