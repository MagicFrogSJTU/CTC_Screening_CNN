''' Created by Yizhi Chen. 20170930'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



# Process images of this size.
SCREEN_VOLUME_SIZE = 48

# Cut volume from original CT data.
CUT_RAW_VOLUME_SIZE = 160



import os.path
import numpy as np
import tensorflow as tf
import os
import polyp_def
import multiprocessing
import SimpleITK
from scipy.interpolate import RegularGridInterpolator as RGI
import time
from scipy.ndimage import binary_dilation
from scipy.ndimage import generate_binary_structure
from multiprocessing import Manager
from screen_test_run import crop
from dataDirectory import DataDirectory
import dataDirectory




class Polyp_Manager:
    def __init__(self):
        self.polyp_list = []
        self.polyps_data_filename = 0
        self.sampleIndex = -1
        self.train_polyp_list = 0
        self.test_polyp_list = 0
        self.cross_validation_fold = 'cross_validation'
        self.dataDirectory = DataDirectory()
        self.data_base_dir = self.dataDirectory.data_base_dir()


    def find_by_index(self, index):
        for polyp in self.polyp_list:
            if polyp.INDEX == index:
                return polyp
        return None


##############################################################################################################
    # TODO:Directory!!
    def train_test_independent_seperation(self, index_of_test, ratio=0.2, ):
        base_dir = self.dataDirectory.independent_dir(num=index_of_test)
        database_dir = self.dataDirectory.data_base_dir()

        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        volume_folds = os.listdir(database_dir)
        num = len(volume_folds)
        chunk_num = num * ratio
        choice = np.array(volume_folds)
        np.random.shuffle(choice)

        f_train = open(os.path.join(base_dir, "trainSet.txt"), "w")
        f_test = open(os.path.join(base_dir, "testSet.txt"), "w")
        f_train_vol = open(os.path.join(base_dir, "trainVolumeRecord.txt"), 'w')
        f_test_vol = open(os.path.join(base_dir, "testVolumeRecord.txt"), 'w')
        for j in range(num):
            volume_dir = os.path.join(database_dir, str(choice[j]))
            for polyp_fold in os.listdir(volume_dir):
                if not os.path.isdir(os.path.join(volume_dir, polyp_fold)):
                    assert polyp_fold == "CT_volume_base_dir.txt"
                    with open(os.path.join(volume_dir, polyp_fold), 'r') as f:
                        line = f.read()
                    if j < (chunk_num):
                        f_test_vol.write(line + "\n")
                    else:
                        f_train_vol.write(line + "\n")
                    continue
                if j < (chunk_num):
                    f_test.write("%d %d \n" % (int(choice[j]), int(polyp_fold)))
                else:
                    f_train.write("%d %d \n" % (int(choice[j]), int(polyp_fold)))
        f_test.close()
        f_train.close()
        f_train_vol.close()
        f_test_vol.close()

##############################################################################################################
    def train_test_cross_validation_seperation(self, pieces=5):
        base_record_dir = os.path.join(self.dataDirectory.base_dir(), self.cross_validation_fold)
        database_dir = self.dataDirectory.data_base_dir()

        if not os.path.exists(base_record_dir):
            os.mkdir(base_record_dir)
        volume_folds = os.listdir(database_dir)
        data_dict = {}
        for volume_fold in volume_folds:
            volume_dir = os.path.join(database_dir, volume_fold)
            with open(os.path.join(volume_dir, "CT_volume_base_dir.txt")) as f:
                data_fold = f.read()
            data_fold = data_fold[36:]
            patient_name = data_fold[:data_fold.find("/")]
            rest_line = data_fold[data_fold.find("/")+1:]
            if "File" in patient_name:
                patient_name += "/" + rest_line[:rest_line.find("/")]
                rest_line = rest_line[rest_line.find("/")+1:]
            if patient_name not in data_dict:
                data_dict[patient_name] = []
            data_dict[patient_name].append([volume_fold, rest_line])

        for patient_name in data_dict:
            if len(data_dict[patient_name]) != 2:
                print(patient_name)
                print(data_dict[patient_name])

        num = len(data_dict)
        random_choice = np.linspace(0,num-1,num)
        np.random.shuffle(random_choice)

        chunk_num = int(num*1.0/pieces)+1
        for i in range(pieces):
            sess_dir = os.path.join(base_record_dir, str(i))
            if not os.path.exists(sess_dir):
                os.mkdir(sess_dir)
            f_train = open(os.path.join(sess_dir,"trainSet.txt"), "w")
            f_test = open(os.path.join(sess_dir, "testSet.txt"), "w")
            f_train_vol = open(os.path.join(sess_dir, "trainVolumeRecord.txt"), 'w')
            f_test_vol = open(os.path.join(sess_dir, "testVolumeRecord.txt"), 'w')
            for j in range(num):
                for index, data_unit in enumerate(data_dict):
                    if index == random_choice[j]:
                        patient_name = data_unit
                        values = data_dict[data_unit]

                for value in values:
                    volume_dir = os.path.join(database_dir, str(value[0]))
                    line = patient_name + '/' + value[1]

                    if j >= (chunk_num * i) and j < ((i + 1) * chunk_num):
                        f_test_vol.write(line+"\n")
                    else:
                        f_train_vol.write(line+"\n")

                    for polyp_fold in os.listdir(volume_dir):
                        if not os.path.isdir(os.path.join(volume_dir, polyp_fold)):
                            assert polyp_fold == "CT_volume_base_dir.txt"
                            continue
                        if j>=(chunk_num*i) and j<((i+1)*chunk_num):
                            f_test.write("%d %d \n" %(int(value[0]), int(polyp_fold)))
                        else:
                            f_train.write("%d %d \n" %(int(value[0]), int(polyp_fold)))
            f_test.close()
            f_train.close()
            f_train_vol.close()
            f_test_vol.close()

####################################################################################################################

    def _bytes_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

####################################################################################################################
    def set_up_polyps_data_base_multiprocessing(self, polyp_dir_list, ):
        '''Multiprocessing Version.'''
        output_dir = self.dataDirectory.data_base_dir()
        cen_index = int(CUT_RAW_VOLUME_SIZE / 2)
        xi, yi, zi = np.mgrid[(-cen_index):cen_index,
                     (-cen_index):cen_index,
                     (-cen_index):cen_index]
        xi = xi.astype(np.float16)
        yi = yi.astype(np.float16)
        zi = zi.astype(np.float16)

        zoomRatios = [2.0, 1.5, 0.75, 0.5, 0.3333]  # Ratio!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        manager = Manager()   # To share the memory.
        coor_dicts = manager.list()
        for ratio in zoomRatios:
            x_zoomed = xi / ratio
            y_zoomed = yi / ratio
            z_zoomed = zi / ratio
            new_coor_dict = {}
            new_coor_dict['zoom_ratio'] = ratio
            new_coor_dict['coor'] = (x_zoomed, y_zoomed, z_zoomed)
            coor_dicts.append(new_coor_dict)
        xi = 0
        yi = 0
        zi = 0

        threads_num = 8
        threads = []

        polyps_thread = [[] for n in range(threads_num)]
        for i, polyp_dirs in enumerate(polyp_dir_list):
            polyps_thread[i % threads_num].append(polyp_dirs)

        polyps_num = []
        sum = 0
        for per_polyps_thread in polyps_thread:
            polyps_num.append(sum)
            sum += len(per_polyps_thread)
        print(polyps_num)

        for i in range(threads_num):
            a = polyps_thread[i]
            t = multiprocessing.Process(target=self.Set_up_Polyps_Data_Base,
                                        args=([a, output_dir, polyps_num[i], coor_dicts]))
            t.daemon = True
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def Set_up_Polyps_Data_Base(self, polyp_dir_list, output_dir, INDEX_Begin=0, coor_dicts=None):
        '''Load polyps data from Original data from Ren Yacheng.
        load volume data. Then load polyps mask data. Then crop polyps CT data from volume data.

        Args:
            polyp_dir_list: list of polyps mask file. [[polyp1 in vol1, polyp2 in vol1], [polyp1 in vol2,...], ...]

        Returns:
            None. Data would be stored in the polyp list of the manager.
        '''
        INDEX = INDEX_Begin  # Variable that counts the ranking number of polyps

        # For each CT volume.
        for i, polyp_dir_vol_unit in enumerate(polyp_dir_list):
            print("%d: %s" % (INDEX, polyp_dir_vol_unit[0]))
            time_begin = time.time()

            output_volume_dir = os.path.join(output_dir, str(INDEX))
            if not os.path.exists(output_volume_dir):
                os.makedirs(output_volume_dir)
            INDEX += 1
            vol_unit = polyp_def.Volume_Data()
            base_dir = polyp_dir_vol_unit[0][0:polyp_dir_vol_unit[0].rfind("/")]
            vol_unit.Set_Directory(base_dir)
            vol_unit.Load_Volume_Data()
            vol_unit.load_colon_mask()

            # make a new file, saving the directory of the base_dir of the
            # volume.
            with open(os.path.join(output_volume_dir, "CT_volume_base_dir.txt"), 'w') as f:
                f.write(base_dir)

            new_polyp_list = []
            # In case there are several polyps in the same area,
            # all masks of polyps should be loaded first.
            # Therefore, in the cropped vol, labels would be correct.
            mask = np.zeros(vol_unit.shape, dtype=np.uint8)
            for j, polyp_dir_unit in enumerate(polyp_dir_vol_unit):

                polyp_unit = polyp_def.Polyp_data()
                polyp_unit.Set_dir(polyp_dir_unit)
                polyp_unit.Load_Dots_Coor()
                new_polyp_list.append(polyp_unit)
                for dot in polyp_unit.dots_list:
                    mask[dot[0], dot[1], dot[2]] = 1

            # Apply colon dilation.
            colon_mask = ((vol_unit.colon_mask + mask)>0.001).astype(np.uint8)
            structure = generate_binary_structure(3, 2)
            colon_mask = binary_dilation(colon_mask, structure, iterations=8).astype(np.uint8)

            vol = vol_unit.rawdata

            # Prepare Function of Interpolation
            x = np.linspace(0, vol.shape[2] - 1, vol.shape[2]).astype(np.float16)
            y = np.linspace(0, vol.shape[1] - 1, vol.shape[1]).astype(np.float16)
            z = np.linspace(0, vol.shape[0] - 1, vol.shape[0]).astype(np.float16)
            func_vol = RGI((z, y, x), vol, bounds_error=False, fill_value=-999)
            func_mask = RGI((z, y, x), mask, bounds_error=False, fill_value=0)
            func_colon_mask = RGI((z, y, x), colon_mask, bounds_error=False, fill_value=0)



            # For each polyp.
            for polyp_index, polyp in enumerate(new_polyp_list):
                if polyp.dots_list == []:
                    print("Dots list is empty!")
                    return 1

                polyp.INDEX = polyp_index

                #################################################################
                # Before all, save basic files of polyp to the disk.
                # Prepare coordinates by adding translation.
                aver = polyp.dots_list.sum(axis=0) / polyp.dots_list.shape[0]
                aver = aver.reshape([3])

                # Save filenames
                left_index = [int(aver[0]-CUT_RAW_VOLUME_SIZE/2),
                              int(aver[1]-CUT_RAW_VOLUME_SIZE/2),
                              int(aver[2]-CUT_RAW_VOLUME_SIZE/2)]
                polyp.polyp_crop_data = crop(vol, CUT_RAW_VOLUME_SIZE,left_index, -999)
                polyp.mask = crop(mask, CUT_RAW_VOLUME_SIZE, left_index, 0)
                assert polyp.mask[int(CUT_RAW_VOLUME_SIZE/2),
                                  int(CUT_RAW_VOLUME_SIZE / 2),
                                  int(CUT_RAW_VOLUME_SIZE / 2)] == 1
                polyp.dilated_colon_mask_data = crop(colon_mask, CUT_RAW_VOLUME_SIZE, left_index, 0)

                polyp.base_folder_dir = os.path.join(output_volume_dir, str(polyp.INDEX))
                polyp.raw_vol_dir = os.path.join(polyp.base_folder_dir, polyp.raw_vol_name)
                polyp.raw_mask_dir = os.path.join(polyp.base_folder_dir, polyp.raw_mask_name)
                polyp.dilated_colon_mask_dir = os.path.join(polyp.base_folder_dir, polyp.dilated_colon_mask_name)
                if not os.path.exists(polyp.base_folder_dir):
                    os.makedirs(polyp.base_folder_dir)

                assert polyp.polyp_crop_data.dtype == np.int16
                assert polyp.mask.dtype == np.uint8
                assert polyp.dilated_colon_mask_data.dtype == np.uint8
                polyp.write_data_into_raw_file()


                #################################################################
                # Start interpolation and file saving
                guess_radius = (polyp.dots_list.shape[0] * 3.0 / 4.0 / np.pi) ** (1 / 3.0)
                #print("guess radius: %s,  INDEX: %d" % (guess_radius, polyp.INDEX))

                writer = tf.python_io.TFRecordWriter(
                            os.path.join(polyp.base_folder_dir, polyp.raw_tf_name))
                example = tf.train.Example(features=tf.train.Features(feature={
                        'volume': self._bytes_feature(polyp.polyp_crop_data.astype(np.float32).tostring()),# Be float32!
                        'mask':self._bytes_feature(polyp.mask.tostring()),}))
                writer.write(example.SerializeToString())
                writer.close()

                writer = tf.python_io.TFRecordWriter(
                            os.path.join(polyp.base_folder_dir, polyp.augment_zoom_tf_name))
                example = tf.train.Example(features=tf.train.Features(feature={
                        'volume': self._bytes_feature(polyp.polyp_crop_data.astype(np.float32).tostring()),
                        'mask':self._bytes_feature(polyp.mask.tostring()),
                    'colon_mask': self._bytes_feature(polyp.dilated_colon_mask_data.tostring())}))
                writer.write(example.SerializeToString())

                for j, coor_dict in enumerate(coor_dicts):
                    zoom_ratio = coor_dict['zoom_ratio']
                    if guess_radius * zoom_ratio < 3:
                        continue
                    if guess_radius * zoom_ratio > 15:
                        continue
                    coor = coor_dict['coor']
                    trans_xi = coor[0] + aver[2]
                    trans_yi = coor[1] + aver[1]
                    trans_zi = coor[2] + aver[0]

                    crop_vol = func_vol((trans_zi, trans_yi, trans_xi))
                    crop_vol = crop_vol.astype(np.float32)

                    crop_mask = func_mask((trans_zi, trans_yi, trans_xi))
                    crop_mask = (crop_mask >= 0.25).astype(np.uint8)

                    crop_colon_mask = func_colon_mask((trans_zi, trans_yi, trans_xi))
                    crop_colon_mask = (crop_colon_mask >= 0.5).astype(np.uint8)

                    if np.sum(crop_colon_mask) < 10:
                        print("%d: %f, %d" % (polyp.dir, zoom_ratio, int(np.sum(crop_colon_mask))))

                    example = tf.train.Example(features=tf.train.Features(feature={
                            'volume': self._bytes_feature(crop_vol.tostring()),
                            'mask': self._bytes_feature(crop_mask.tostring()),
                            'colon_mask': self._bytes_feature(crop_colon_mask.tostring()),
                    }))
                    writer.write(example.SerializeToString())

                    '''
                    # test
                    print(polyp.INDEX)
                    vol = SimpleITK.GetImageFromArray(np.transpose(crop_vol,(2,1,0)))
                    SimpleITK.WriteImage(vol, os.path.join(polyp.base_folder_dir, "polyp" + str(j) + ".nii.gz"))
                    mask = SimpleITK.GetImageFromArray(np.transpose(crop_mask,(2,1,0)))
                    SimpleITK.WriteImage(mask, os.path.join(polyp.base_folder_dir, "mask" + str(j) + ".nii.gz"))
                    zoom_colon_mask = SimpleITK.GetImageFromArray(np.transpose(crop_colon_mask,(2,1,0)))
                    SimpleITK.WriteImage(zoom_colon_mask,
                                         os.path.join(polyp.base_folder_dir, "colon_mask" + str(j) + ".nii.gz"))

                    '''
                writer.close()
            print("Time consumed: %f" %(time.time()-time_begin))






    def read_polyps_from_disk(self, record_file_dir=None, whichone=None):
        '''Read polyps data in 1,2,3...directory.
        Consist of CT data of polyp in size [CUT_RAW_VOLUME_SIZE,CUT_RAW_VOLUME_SIZE,CUT_RAW_VOLUME_SIZE],
        and mask data in the same size.
        Args:
            dir: the directory of the main folder
        '''
        if record_file_dir is None:
            raise IOError
        self.polyp_list = []
        index_list = []
        TEST_RECORD_DIRCTORY = os.path.join(record_file_dir, "testSet.txt")
        TRAIN_RECORD_DIRECTORY = os.path.join(record_file_dir, "trainSet.txt")
        if whichone == 'train':
            with open(TRAIN_RECORD_DIRECTORY, 'r') as f:
                for line in f.readlines():
                    words = line[:-1].split()
                    index_list.append([int(words[0]),int(words[1])])
        elif whichone == 'test':
            with open(TEST_RECORD_DIRCTORY, 'r') as f:
                for line in f.readlines():
                    words = line[:-1].split()
                    index_list.append([int(words[0]),int(words[1])])
        elif whichone == 'all':
            with open(TRAIN_RECORD_DIRECTORY, 'r') as f:
                for line in f.readlines():
                    words = line[:-1].split()
                    index_list.append([int(words[0]),int(words[1])])
            with open(TEST_RECORD_DIRCTORY, 'r') as f:
                for line in f.readlines():
                    words = line[:-1].split()
                    index_list.append([int(words[0]),int(words[1])])
        else:
            index_list.append(whichone)

        for index_unit in index_list:
            unit = polyp_def.Polyp_data()
            unit.INDEX = [index_unit[0], index_unit[1]]
            unit.base_folder_dir = os.path.join(self.data_base_dir, str(index_unit[0]), str(index_unit[1]))
            unit.raw_vol_dir = os.path.join(unit.base_folder_dir, unit.raw_vol_name)
            unit.raw_mask_dir = os.path.join(unit.base_folder_dir, unit.raw_mask_name)
            unit.dilated_colon_mask_dir = os.path.join(unit.base_folder_dir, unit.dilated_colon_mask_name)
            unit.read_data_from_raw_file()
            self.polyp_list.append(unit)

        print("Num of polyps loaded: ,", len(self.polyp_list))
        print("Polyps loading finished!")




    def data_base_set_up_from_dir(self):
        '''Loop through the directory, find polyp_dots file.
        '''
        base_dir = self.dataDirectory.raw_CTdata_dir()
        polylist = []
        size = 0
        for dirpath, dirnames, filenames in os.walk(base_dir):
            per_polylist = []
            for i, file in enumerate(filenames):
                if file.find("poly_dots")!= -1:
                    per_polylist.append(os.path.join(dirpath, file))
                    size += 1
            if per_polylist != []:
                polylist.append(per_polylist)

        print("CT Volume Data Size: ", size)
        return polylist



class Volume_Manager:

    def get_volume_from_record(self, record_url, base_url):
        with open(record_url, 'r') as f:
            lines = f.readlines()
        self.volume_list = []
        for line in lines:
            line = line[:-1]
            line = base_url + "/SegmentedColonData/" + line
            print(line)
            for file in os.listdir(line):
                if file == 'oriInterpolatedCTData.raw' or file == 'InterpolatedCTData.raw':
                    new_volume = polyp_def.Volume_Data()
                    new_volume.Set_Directory(line)
                    self.volume_list.append(new_volume)
                    break


    def get_volume_from_main_directory(self, record_url):
        self.volume_list = []
        for dirpath, dirnames, filenames in os.walk(record_url):
            for filename in filenames:
                if filename == 'oriInterpolatedCTData.raw' or filename == 'InterpolatedCTData.raw':
                    new_volume = polyp_def.Volume_Data()
                    new_volume.Set_Directory(dirpath)
                    self.volume_list.append(new_volume)
                    break

    def calculate_colon_mask_dilation(self):
        for volume in self.volume_list:
            a=time.time()
            volume.Load_Volume_Data()
            volume.load_colon_mask()
            volume.colon_mask_dilation()
            image = SimpleITK.GetImageFromArray(volume.dilated_colon_mask)
            SimpleITK.WriteImage(image, os.path.join(volume.base_dir, "dilated_colon_mask.nii.gz"))
            volume.clear_volume_data()
            print("time consumed:", time.time()-a)


if __name__ == '__main__':

    #'''
    # Load from dicom files and save.
    if 0:
        polyp_manager = Polyp_Manager()
        polyp_list = polyp_manager.data_base_set_up_from_dir()
        polyp_manager.set_up_polyps_data_base_multiprocessing(polyp_list)
    #'''


    # Generate train set and test test.
    if 1:
        polyp_manager = Polyp_Manager()
        #polyp_manager.train_test_independent_seperation()
        polyp_manager.train_test_cross_validation_seperation()
        #polyp_manager.train_test_independent_seperation(index_of_test=1, ratio=0.2)

    if 0:
        polyp_manager = Polyp_Manager()
        polyp_manager.read_polyps_from_disk(whichone='test')
        for polyp in polyp_manager.polyp_list:
            print(polyp.INDEX, np.sum(polyp.mask[56:104,56:104,56:104]))
    if 0:
        volume_manager = Volume_Manager()
        dataDir = dataDirectory.DataDirectory()
        volume_record_dir = os.path.join(dataDir.get_current_record_dir(), "testVolumeRecord.txt")
        volume_manager.get_volume_from_record(volume_record_dir)
        volume_manager.calculate_colon_mask_dilation()




