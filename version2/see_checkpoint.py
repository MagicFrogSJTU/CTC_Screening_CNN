#!/usr/bin/env python
# encoding: utf-8

from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file("cross_0/Screen/Train/model.ckpt-20000",
        tensor_name='',
        all_tensors=False)

