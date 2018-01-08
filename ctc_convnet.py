# forked and revised by cyz. 20171004


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctc_input
import tensorflow as tf
import numpy as np
import scipy.ndimage
import ctc_input_cpu

def dialate(labels):
    a = labels.astype(np.uint8)
    for i in range(labels.shape[0]):
        a[i,:,:,:,0] = scipy.ndimage.binary_dilation(a[i,:,:,:,0], scipy.ndimage.generate_binary_structure(3,2), 1)
    a = a.astype(labels.dtype)

    return a



def train(total_loss, global_step, Parameters):
    """Train CIFAR-10 dice_typeB_0.01_2res_2slices_2.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
        train_op: op for training.
    """
    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    lr = tf.train.exponential_decay(Parameters.INITIAL_LEARNING_RATE, global_step, 1000, 0.95, name='exponential_decay')
    tf.summary.scalar('learning rate', lr)

    opt = tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(Parameters.MOVING_AVERAGE, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies([variable_averages_op, apply_gradient_op,]+ update_ops):
        train_op = tf.no_op(name='train')
    return train_op


def inputs(eval_data, record_file_dir, database_dir, Parameters):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
    """
    if eval_data:
        if Parameters.ROTATE_GPU:
            print("GPU ROTATION")
            images, labels = ctc_input.inputs(eval_data, Parameters.BATCH_SIZE, record_file_dir, database_dir,
                                          Parameters)
        else:
            print("CPU ROTATION")
            images, labels = ctc_input_cpu.inputs(eval_data, Parameters.BATCH_SIZE, record_file_dir, database_dir,
                                          Parameters)
            
    else:
        images, labels = ctc_input.inputs(eval_data, 1, record_file_dir, database_dir,Parameters)

    return images, labels
