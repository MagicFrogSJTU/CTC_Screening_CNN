from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class PARAMETERS:
    MOVING_AVERAGE = 0.999
    INITIAL_LEARNING_RATE = 0.0001  #1e-4
    TYPEA = 0
    TYPEB = 0.001
    TRUE_SAMPLE_RATIO = 0.5
    IF_CONTINUE = 1

    LOG_FREQUENCY = 20
    LOG_DEVICE_PLACEMENT = False
    MAX_STEPS = 10000
    BATCH_SIZE = 16
Parameters = PARAMETERS()


import sys
import os
sys.path.append(os.path.join(os.getcwd(),".."))
import tensorflow as tf
from dataDirectory import DataDirectory
from screen_cnn import inference
from ctc_screen_train import train as screen_train



def getloss(logits, labels, scope):
    num_of_labels_vol = tf.reduce_sum(labels, [1, 2, 3, 4])
    true_batch = tf.cast(num_of_labels_vol > 0.5, tf.float32)
    num_of_true_examples = tf.reduce_sum(true_batch)
    num_of_false_examples = Parameters.BATCH_SIZE- num_of_true_examples
    select = (500 / ((num_of_labels_vol+1e-5)**0.5 + 0.01)) * true_batch
    select = select / tf.reduce_sum(select) * num_of_true_examples #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

    logits2 = tf.reduce_sum(logits*logits, axis=[1,2,3,4])
    dice_loss = 1 - (2*tf.reduce_sum(logits*labels, axis=[1,2,3,4])+Parameters.TYPEA)/\
                (logits2 + tf.reduce_sum(labels*labels, axis=[1,2,3,4])+0.000001)
    dice_loss = dice_loss * select
    with tf.name_scope("dice_loss_mean"):
        dice_loss_mean = tf.reduce_sum(dice_loss) / (num_of_true_examples+1e-6)


    logits2_big = tf.reduce_sum(((logits-0.5)*tf.cast(logits>0.5, tf.float32))**2, axis=[1,2,3,4])
    logits2_zero = logits2_big * tf.cast(num_of_labels_vol<0.5, tf.float32)
    with tf.name_scope("zero_loss_mean"):
        zero_loss_mean = tf.reduce_sum(logits2_zero) * Parameters.TYPEB / (num_of_false_examples+1e-6)

    tf.add_to_collection('losses', zero_loss_mean)
    tf.add_to_collection('losses', dice_loss_mean)
    losses = tf.get_collection('losses', scope)
    regular_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_losses = tf.add_n(losses + regular_losses, name='total_loss')
    for l in losses + regular_losses + [total_losses]:
        tf.summary.scalar(l.op.name, l)

    return total_losses




def main(argv=None):  # pylint: disable=unused-argument
    dataDirectory = DataDirectory()
    database_dir = dataDirectory.data_base_dir()
    train_dir = os.path.join(dataDirectory.get_current_model_dir(),
                             dataDirectory.checkpoint_fold)
    record_file_dir = dataDirectory.get_current_train_record_dir()

    IF_CONTINUE_TRAIN = Parameters.IF_CONTINUE == 1
    if (not IF_CONTINUE_TRAIN) and tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)
    print("checkpoint directory:", train_dir)
    print("record directory:", record_file_dir)

    screen_train(train_dir, record_file_dir, database_dir, inference, getloss, Parameters)



if __name__ == '__main__':
    tf.app.run()