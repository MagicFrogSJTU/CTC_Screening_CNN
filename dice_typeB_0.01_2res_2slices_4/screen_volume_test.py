import sys
import os
sys.path.append(os.getcwd()+"/..")

from dataDirectory import DataDirectory
from screen_volume import main
from screen_volume import Test
from model_train import Parameters
from screen_cnn import inference

if __name__ == '__main__':
    directory = DataDirectory()
    checkpoint_dir = directory.get_current_checkpoint_dir()
    record_dir = directory.get_current_record_dir()

    #main(checkpoint_dir, record_dir, 'testVolumeRecord.txt', inference, Parameters)

    Test(0.99, record_dir, ifwrite=True)
