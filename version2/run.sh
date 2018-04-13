#!/bin/bash
FOLD=2
echo $FOLD
python train.py --fold=$FOLD
python screen_test.py --fold=$FOLD
python screen.py --fold=$FOLD
