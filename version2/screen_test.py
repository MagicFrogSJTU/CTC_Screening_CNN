import sys
import os
sys.path.append(os.path.join(os.getcwd(),".."))
from dataDirectory import DataDirectory
import data_input
from screen_test_run import main
from screen_test_run import Test
from train import Parameters
from network import inference
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, help='which cross validation fold')
FLAGS = parser.parse_args()

if __name__ == '__main__':
    dataDirectory = DataDirectory()
    dataDirectory.cross_index = FLAGS.fold
    train_dir = os.path.join(dataDirectory.get_current_model_dir(),
                             dataDirectory.checkpoint_fold)
    record_dir = dataDirectory.get_current_record_dir()

    polyp_manager = data_input.Polyp_Manager()
    polyp_manager.read_polyps_from_disk(record_dir, 'test')

    #main(train_dir, record_dir, 'test', inference, Parameters)

    Test(polyp_manager, 0.9999)
    Test(polyp_manager, 0.999)
    Test(polyp_manager, 0.99, ifwrite=False)
    Test(polyp_manager, 0.9)
    #Test(polyp_manager, 0.93)
    #Test(polyp_manager, 0.95)
    #Test(polyp_manager, 0.97)
    #Test(polyp_manager, 0.993)
    #Test(polyp_manager, 0.6)
    Test(polyp_manager, 0.5)
