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
from screen_cnn import inference
from model_train import Parameters
tf.logging.set_verbosity(tf.logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Parameters.eval_interval_secs=120
Parameters.run_once=False


def main(argv=None):
    dataDirectory = DataDirectory()
    model_dir = dataDirectory.get_current_model_dir()
    eval_dir = os.path.join(dataDirectory.get_current_model_dir(),
                             dataDirectory.eval_fold)
    record_file_dir = dataDirectory.get_current_test_record_dir()
    if tf.gfile.Exists(eval_dir):
        tf.gfile.DeleteRecursively(eval_dir)
    tf.gfile.MakeDirs(eval_dir)
    print("dice_typeB_0.01_2res_2slices_2 directory:", model_dir)
    print("record directory:", record_file_dir)
    evaluate(model_dir, record_file_dir, dataDirectory.data_base_dir(), inference, Parameters)


if __name__ == '__main__':
    tf.app.run()
