import sys
import os
sys.path.append(os.getcwd()+"/..")
import screen_volume
from train import ModelConfig
from network import inference
from dataBase import DataBase

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, help='which cross validation fold')
FLAGS = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    cfg = ModelConfig(FLAGS.fold)
    db = DataBase(FLAGS.fold)
    cfg.multi_flag = [1,0]
    print('multi_flag: ', cfg.multi_flag)
    cfg.result_file_fold = str(cfg.cross_index)

    screen_volume.screen_cnn(inference, db, cfg)
    screen_volume.analysis_of_screen(db, cfg, 0.999, 0.5)
    #screen_volume.produce_tf_samples(volume_manager, multi_flag, result_file_fold)
