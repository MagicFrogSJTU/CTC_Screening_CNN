import sys
import os
sys.path.append(os.getcwd()+"/..")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    multi_flag = [1,0]
    print('multi_flag: ', multi_flag)
    volume_manager = Volume_Manager()
    volume_manager.get_volume_from_record(record_dir)


    screen_volume.screen_cnn(checkpoint_dir, volume_manager, inference, Parameters, multi_flag)
    screen_volume.analysis_of_screen(volume_manager, 0.9, 0.9, multi_flag, )
    #screen_volume.produce_tf_samples(volume_manager, multi_flag)
