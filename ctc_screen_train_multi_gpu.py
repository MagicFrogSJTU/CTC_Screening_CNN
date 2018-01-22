from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import time
import tensorflow as tf
import ctc_convnet




def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train(train_dir, record_file_dir, database_dir, inference, getloss, Parameters):
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as graph, tf.device('/cpu:0'):
        global_step = tf.train.get_or_create_global_step()

        volumes, labels = ctc_convnet.inputs(True, record_file_dir, database_dir, Parameters)
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [volumes, labels], capacity=2*Parameters.NUM_GPUS)
        tower_grads = []

        decay_learning_rate = tf.train.exponential_decay(Parameters.INITIAL_LEARNING_RATE, global_step,
                                                         decay_steps=1000, decay_rate=0.95)
        opt = tf.train.AdamOptimizer(decay_learning_rate)

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(1, Parameters.NUM_GPUS+1):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('gpu', i)) as scope:
                        vol_batch, label_batch = batch_queue.dequeue()

                        logits =inference(vol_batch, True)
                        loss = getloss(logits, label_batch, scope)

                        tf.get_variable_scope().reuse_variables()
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
                        with tf.control_dependencies(update_ops):
                            grads = opt.compute_gradients(loss)

                        tower_grads.append(grads)

        grads = average_gradients(tower_grads)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        variable_averages = tf.train.ExponentialMovingAverage(Parameters.MOVING_AVERAGE, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([variable_averages_op, apply_gradient_op]):
            train_op = tf.no_op(name='train')

        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=Parameters.LOG_DEVICE_PLACEMENT)
        config.gpu_options.allow_growth = True

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % Parameters.LOG_FREQUENCY== 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = (Parameters.NUM_GPUS)*Parameters.LOG_FREQUENCY*Parameters.BATCH_SIZE/ duration
                    sec_per_batch = float(duration /Parameters.LOG_FREQUENCY/ (Parameters.NUM_GPUS))

                    format_str = ('%s: step %d, loss = %.3f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        class TraceHook(tf.train.SessionRunHook):
            """Hook to perform Traces every N steps."""

            def __init__(self, ckptdir, every_step=50, trace_level=tf.RunOptions.FULL_TRACE):
                self._trace = every_step == 1
                self.writer = tf.summary.FileWriter(ckptdir)
                self.trace_level = trace_level
                self.every_step = every_step

            def begin(self):
                self._global_step_tensor = tf.train.get_global_step()
                if self._global_step_tensor is None:
                    raise RuntimeError("Global step should be created to use _TraceHook.")

            def before_run(self, run_context):
                if self._trace:
                    options = tf.RunOptions(trace_level=self.trace_level)
                else:
                    options = None
                return tf.train.SessionRunArgs(fetches=self._global_step_tensor,
                                               options=options)

            def after_run(self, run_context, run_values):
                global_step = run_values.results - 1
                if self._trace:
                    self._trace = False
                    self.writer.add_run_metadata(run_values.run_metadata,
                                                 'global_step%d'%global_step)
                if not (global_step + 1) % self.every_step:
                    self._trace = True

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=Parameters.MAX_STEPS),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook(),],
                #TraceHook(train_dir)],
                save_checkpoint_secs=60,
                config=config
        ) as mon_sess:
            graph.finalize()
            while not mon_sess.should_stop():
                mon_sess.run(train_op)



