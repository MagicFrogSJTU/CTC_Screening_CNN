#!/bin/bash
FOLD=0
echo $FOLD
python train.py --fold=$FOLD
python screen_test.py --fold=$FOLD
python screen.py --fold=$FOLD
