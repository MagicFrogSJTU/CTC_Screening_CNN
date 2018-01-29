"""Created by Yizhi Chen. 20171008
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
#sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),".."))
import tensorflow as tf
from dataDirectory import DataDirectory
from ctc_screen_eval import evaluate
from network import inference
from train import Parameters
tf.logging.set_verbosity(tf.logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Parameters.eval_interval_secs=180
Parameters.run_once=False

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, help='which cross validation fold')
FLAGS = parser.parse_args()

def main(argv=None):
    dataDirectory = DataDirectory()
    dataDirectory.cross_index = FLAGS.fold
    model_dir = dataDirectory.get_current_model_dir()
    eval_dir = os.path.join(dataDirectory.get_current_model_dir(),
                             dataDirectory.eval_fold)
    if tf.gfile.Exists(eval_dir):
        tf.gfile.DeleteRecursively(eval_dir)
    tf.gfile.MakeDirs(eval_dir)
    evaluate(model_dir, dataDirectory, inference, Parameters)


if __name__ == '__main__':
    tf.app.run()
