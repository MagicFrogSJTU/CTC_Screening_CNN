import sys
import os
sys.path.append(os.getcwd()+"/..")
from dataDirectory import DataDirectory
import screen_volume
from train import Parameters
from network import inference
from data_input import Volume_Manager

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, help='which cross validation fold')
FLAGS = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    directory = DataDirectory()
    directory.cross_index = FLAGS.fold
    checkpoint_dir = directory.get_current_checkpoint_dir()
    record_dir = os.path.join(directory.get_current_record_dir(),
                              'testVolumeRecord.txt')
    print(checkpoint_dir)
    print(record_dir)
    multi_flag = [1,0]
    print('multi_flag: ', multi_flag)
    volume_manager = Volume_Manager()
    base_dir = directory.base_dir()
    volume_manager.get_volume_from_record(record_dir, base_dir)

    result_file_fold = str(directory.cross_index)
    #screen_volume.screen_cnn(checkpoint_dir, volume_manager, inference, Parameters, multi_flag, result_file_fold)
    #screen_volume.analysis_of_screen(volume_manager, 0.999, 0.99, multi_flag, result_file_fold)
    screen_volume.produce_tf_samples(volume_manager, multi_flag, result_file_fold)
