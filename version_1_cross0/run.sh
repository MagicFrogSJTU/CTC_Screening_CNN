#!/bin/bash
FOLD=2
echo $FOLD
python model_train.py --fold=$FOLD
#python screen_test.py --fold=$FOLD
python screen_volume_test.py --fold=$FOLD
