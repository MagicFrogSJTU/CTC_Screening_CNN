import sys
import os
sys.path.append(os.getcwd()+"/..")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from dataDirectory import DataDirectory
import screen_volume
from model_train import Parameters
from screen_cnn import inference
from data_input import Volume_Manager
if __name__ == '__main__':
    directory = DataDirectory()
    checkpoint_dir = directory.get_current_checkpoint_dir()
    record_dir = os.path.join(directory.get_current_record_dir(),
                              'testVolumeRecord.txt')

    volume_manager = Volume_Manager()
    volume_manager.get_volume_from_record(record_dir)


    #screen_volume.screen_cnn(checkpoint_dir, volume_manager, inference, Parameters)
    screen_volume.analysis_of_screen(volume_manager, seed_threshold=0.95, grow_threshold=0.9, )
