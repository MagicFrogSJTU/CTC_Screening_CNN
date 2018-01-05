
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import time
import tensorflow as tf
import ctc_convnet




def train(train_dir, record_file_dir, database_dir, inference, getloss, Parameters):
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        with tf.device('/gpu:0'):
            volumes, labels = ctc_convnet.inputs(True, record_file_dir, database_dir, Parameters)

        with tf.name_scope('%s_%d' % ('gpu', 0)) as scope, tf.device('/gpu:1'):
            logits = inference(volumes, True)
            loss = getloss(logits, labels, scope)
            train_op = ctc_convnet.train(loss, global_step, Parameters)

        config = tf.ConfigProto(allow_soft_placement=True)
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
                if self._step % Parameters.LOG_FREQUENCY == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = Parameters.LOG_FREQUENCY * Parameters.BATCH_SIZE/ duration
                    sec_per_batch = float(duration / Parameters.LOG_FREQUENCY)

                    format_str = ('%s: step %d, loss = %.3f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=Parameters.MAX_STEPS),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                save_checkpoint_secs=60,
                config=config
        ) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


