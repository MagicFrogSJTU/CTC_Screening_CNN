#!/bin/bash
python model_train_multi_gpu.py --fold=0
python screen_test.py --fold=0
python screen_volume_test.py --fold=0
