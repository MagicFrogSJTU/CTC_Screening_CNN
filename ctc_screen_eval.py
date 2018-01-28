"""Created by Yizhi Chen. 20171008
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import ctc_convnet
import os
from dataDirectory import DataDirectory
tf.logging.set_verbosity(tf.logging.INFO)

batch_size=1

def eval_once(saver, checkpoint_dir, number_of_samples, summary_writer, summary_op, size_of_true_positive1, size_of_false_positive1,
                      size_of_true_positive2, size_of_false_positive2, size_of_true_positive3, size_of_false_positive3,
                      size_of_positive, dice_loss):
    """Run Eval once.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config,) as sess:
        sess.run(tf.local_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/dice_typeB_0.01_2res_2slices_2.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            size_true_total1 = 0
            size_false_total1 = 0
            size_true_total2 = 0
            size_false_total2 = 0
            size_true_total3 = 0
            size_false_total3 = 0
            step = 0
            AtLeastOneDot1 = 0.0
            AtLeastOneDot2 = 0.0
            AtLeastOneDot3 = 0.0
            dice_loss_total = 0.0


            while step < number_of_samples and not coord.should_stop():
                [size_true1, size_false1, size_true2, size_false2,
                 size_true3, size_false3, size, dice_loss_computed] = sess.run([size_of_true_positive1, size_of_false_positive1,
                      size_of_true_positive2, size_of_false_positive2, size_of_true_positive3, size_of_false_positive3,
                                                               size_of_positive, dice_loss])
                #print("%d: %f, %f, %f" %(step, size_true2, size_false2, size_total))
                size_true_total1 += size_true1*1.0/size
                size_false_total1 += size_false1*1.0/size
                size_true_total2 += size_true2*1.0/size
                size_false_total2 += size_false2*1.0/size
                size_true_total3 += size_true3*1.0/size
                size_false_total3 += size_false3*1.0/size
                dice_loss_total += dice_loss_computed
                step += 1
                if size_true1>0:
                    AtLeastOneDot1+=1.0
                if size_true2>0:
                    AtLeastOneDot2+=1.0
                if size_true3>0:
                    AtLeastOneDot3+=1.0

            # Compute precision @ 1.
            ratio_true_aver1 = size_true_total1 / step
            ratio_false_aver1 = size_false_total1 / step
            ratio_atleastonedot1 = AtLeastOneDot1 / step
            ratio_true_aver2 = size_true_total2 / step
            ratio_false_aver2 = size_false_total2 / step
            ratio_atleastonedot2 = AtLeastOneDot2 / step
            ratio_true_aver3 = size_true_total3 / step
            ratio_false_aver3 = size_false_total3 / step
            ratio_atleastonedot3 = AtLeastOneDot3 / step
            dice_loss_aver = dice_loss_total / step
            print(global_step)
            print("the ratios of 0.1 is %.3f, %.3f, %.3f" %(ratio_true_aver1, ratio_false_aver1, ratio_atleastonedot1))
            print("the ratios of 0.5 is %.3f, %.3f, %.3f" %(ratio_true_aver2, ratio_false_aver2, ratio_atleastonedot2))
            print("the ratios of 0.9 is %.3f, %.3f, %.3f" %(ratio_true_aver3, ratio_false_aver3, ratio_atleastonedot3))
            print(dice_loss_aver)
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate(model_dir, record_dir, database_dir, inference, Parameters):
    """Eval for a number of steps."""
    number_of_samples = 0
    with open(record_dir, 'r') as f:
        number_of_samples = len(f.readlines())
    print("number of samples:", number_of_samples)

    dataDirectory = DataDirectory()
    eval_dir = os.path.join(model_dir, dataDirectory.eval_fold)
    checkpoint_dir = os.path.join(model_dir, dataDirectory.checkpoint_fold)
    with tf.Graph().as_default()as g:
        with tf.device('/cpu:0'):
            volumes, labels = ctc_convnet.inputs(False, record_dir, database_dir, Parameters)


        # Build a Graph that computes the logits predictions from the
        # inference dice_typeB_0.01_2res_2slices_2.
        logits = inference(volumes, False)

        logits2 = tf.reduce_sum(logits*logits, axis=[1,2,3,4])
        dice_loss = 1 - 2*tf.reduce_sum(logits*labels, axis=[1,2,3,4])/\
                (logits2 + tf.reduce_sum(labels*labels, axis=[1,2,3,4])+0.000001)

        logits_reshape = tf.reshape(logits, [int(logits.shape[1]*logits.shape[2]*logits.shape[3]*batch_size)])
        labels_reshape = tf.reshape(labels, [int(labels.shape[1]*labels.shape[2]*labels.shape[3]*batch_size)])

        real1 = tf.cast(logits_reshape>0.1, tf.float32)
        real2 = tf.cast(logits_reshape > 0.5, tf.float32)
        real3 = tf.cast(logits_reshape > 0.9, tf.float32)

        size_of_positive = tf.reduce_sum(labels_reshape)


        true_positive1 = tf.reduce_sum(real1*labels_reshape) # Onlyo labels of polyp remains.
        size_of_true_positive1 = tf.reduce_sum(true_positive1)
        #size_of_positive = tf.Print(size_of_positive,[size_of_positive])
        size_of_predict1 = tf.reduce_sum(real1)
        size_of_false_positive1 = tf.subtract(size_of_predict1, size_of_true_positive1)

        true_positive2 = tf.reduce_sum(real2 * labels_reshape)  # Onlyo labels of polyp remains.
        size_of_true_positive2 = tf.reduce_sum(true_positive2)
        # size_of_positive = tf.Print(size_of_positive,[size_of_positive])
        size_of_predict2 = tf.reduce_sum(real2)
        size_of_false_positive2 = tf.subtract(size_of_predict2, size_of_true_positive2)

        true_positive3 = tf.reduce_sum(real3 * labels_reshape)  # Onlyo labels of polyp remains.
        size_of_true_positive3 = tf.reduce_sum(true_positive3)
        # size_of_positive = tf.Print(size_of_positive,[size_of_positive])
        size_of_predict3 = tf.reduce_sum(real3)
        size_of_false_positive3 = tf.subtract(size_of_predict3, size_of_true_positive3)


        variable_averages = tf.train.ExponentialMovingAverage(Parameters.MOVING_AVERAGE)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(eval_dir, g)

        while True:
            eval_once(saver, checkpoint_dir, number_of_samples, summary_writer, summary_op, size_of_true_positive1, size_of_false_positive1,
                      size_of_true_positive2, size_of_false_positive2, size_of_true_positive3, size_of_false_positive3,
                      size_of_positive, dice_loss)
            if Parameters.run_once:
                break
            time.sleep(Parameters.eval_interval_secs)





