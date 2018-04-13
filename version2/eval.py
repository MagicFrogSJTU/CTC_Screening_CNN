"""Created by Yizhi Chen. 20171008
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
sys.path.append(os.path.join(os.getcwd(),".."))
import tensorflow as tf
from ctc_screen_eval import evaluate
from network import inference
tf.logging.set_verbosity(tf.logging.INFO)
from train import ModelConfig
from dataBase import DataBase

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=None, help='which cross validation fold')
FLAGS = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class EvalModelConfig(ModelConfig):
    EVAL_INTERVAL_SECS = 180
    RUN_ONCE = False

    def __init__(self, whichfold=None):
        assert whichfold is not None
        self.cross_index = whichfold
        print("Nested Cross Validation: Under fold:", self.cross_index)
        #TODO
        # super(EvalModelConfig, self).__init__(whichfold)

def main(argv=None):
    db = DataBase(whichfold=FLAGS.fold)
    cfg = EvalModelConfig(FLAGS.fold)

    eval_dir = cfg.get_current_eval_dir()
    if tf.gfile.Exists(eval_dir):
        tf.gfile.DeleteRecursively(eval_dir)
    tf.gfile.MakeDirs(eval_dir)
    evaluate(inference, cfg, db)


if __name__ == '__main__':
    tf.app.run()
