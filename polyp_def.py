import numpy as np
import dicom
import data_input
import tensorflow as tf
import os
import SimpleITK
import time
from scipy.ndimage import binary_dilation
from scipy.ndimage import generate_binary_structure
CUT_RAW_VOLUME_SIZE = data_input.CUT_RAW_VOLUME_SIZE
SCREEN_VOLUME_SIZE = data_input.SCREEN_VOLUME_SIZE
SCREEN_OUTPUT_SIZE = data_input.SCREEN_VOLUME_SIZE

class Polyp_data:
    '''data object for polyp'''
    polyp_dir_name = "polyp_dots_dir.txt"
    raw_vol_name = "vol.nii.gz"
    raw_mask_name = "mask.nii.gz"
    colon_mask_name = "colon_mask.nii.gz"
    dilated_colon_mask_name = "dilated_colon_mask.nii.gz"
    augment_rt_tf_name_prefix = "augment_data"
    augment_rt_tf_name = augment_rt_tf_name_prefix+".tf"
    augment_zoom_tf_name_prefix = "augment_zoom"
    augment_zoom_tf_name = augment_zoom_tf_name_prefix + ".tf"
    raw_tf_name = "raw_data.tf"                              # vol: float32, mask: uint8


    def __init__(self):
        self.INDEX = -1
        self.dots_list = []
        self.dir = 0
        self.base_dir = 0


        # directory would be accessed when read_data_from_raw_file() or write
        self.base_folder_dir = 0

        self.raw_vol_dir = 0
        self.raw_mask_dir = 0
        self.colon_mask_dir = 0
        self.dilated_colon_mask_dir = 0
        self.augment_RT_vol_dir = 0
        self.augment_RT_mask_dir = 0

        # variables containing 3D data.
        self.polyp_crop_data = 0             # int16
        self.colon_mask_crop_data = []        # uint8
        self.mask = 0                        # uint8
        self.augment_RT_vol_data = 0         # float16
        self.augment_RT_mask_data = 0        # uint8


    def Set_dir(self, directory):
        self.dir = directory
        self.base_dir = self.dir[0:self.dir.rfind("/")]


    def Load_Dots_Coor(self):
        self.dots_list = []
        try:
            f = open(self.dir)
            lines = f.readlines()
            self.dots_list = np.zeros((len(lines), 3), dtype=np.int16)

            # The dimension of no.array is inverse. [c,b,a]
            for i, line in enumerate(lines):
                num_str = line[0:line.find(",")]
                self.dots_list[i, 2] = int(num_str)
                num_str = line[line.find(",")+1:line.rfind(",")]
                self.dots_list[i, 1] = int(num_str)
                num_str = line[line.rfind(",")+1:]
                self.dots_list[i, 0] = int(num_str)
            f.close()
        except Exception as e:
            print(type(e))
            print("Wrong reading polyp dots data")

    def Crop_Polyp_CTData(self, vol_unit, mask, size):
        '''size must be 1d. homogeneous'''
        # Compute center point of dots
        if self.dots_list==[]:
            print("Dots list is empty!")
            return 1

        volume_data = vol_unit.rawdata
        colon_mask = vol_unit.colon_mask

        self.polyp_crop_data = -999 * np.ones((size,size,size),
                                       dtype=np.int16)
        self.colon_mask_crop_data = np.zeros((size,size,size),
                                             dtype=np.uint8)
        self.mask = np.zeros((size,size,size),
                            dtype=np.uint8)
        aver = self.dots_list.sum(axis=0)/self.dots_list.shape[0]
        aver = aver.reshape([3])
        #print "aver: ", aver
        copy_ini = aver - size/2
        copy_end = aver + (size - size/2)
        paste_ini = [0,0,0]
        paste_end = [size, size, size]
        for i in range(0,3):
            if copy_ini[i] < 0:
                paste_ini[i] = 0 - copy_ini[i]
                copy_ini[i] = 0
                pass
            elif copy_end[i] > volume_data.shape[i]:
                paste_end[i] = size - (copy_end[i]-volume_data.shape[i])
                copy_end[i] = volume_data.shape[i]

        #dot_ini = copy_end.astype(np.int)
        #dot_end = copy_ini.astype(np.int)
        self.polyp_crop_data[paste_ini[0]:paste_end[0],
                             paste_ini[1]:paste_end[1],
                             paste_ini[2]:paste_end[2]] = \
            volume_data[copy_ini[0]:copy_end[0],
                        copy_ini[1]:copy_end[1],
                        copy_ini[2]:copy_end[2]]

        self.colon_mask_crop_data[paste_ini[0]:paste_end[0],
                             paste_ini[1]:paste_end[1],
                             paste_ini[2]:paste_end[2]] = \
            colon_mask[copy_ini[0]:copy_end[0],
                        copy_ini[1]:copy_end[1],
                        copy_ini[2]:copy_end[2]]

        # Generate mask, which corresponds to the crop data.
        self.mask[paste_ini[0]:paste_end[0],
                  paste_ini[1]:paste_end[1],
                  paste_ini[2]:paste_end[2]] = \
             mask[copy_ini[0]:copy_end[0],
                  copy_ini[1]:copy_end[1],
                  copy_ini[2]:copy_end[2]]

    def _bytes_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    def save_vol_mask_into_tf_file(self):
        writer = tf.python_io.TFRecordWriter(os.path.join(self.base_folder_dir, self.raw_data_name))
        example = tf.train.Example(features=tf.train.Features(feature={
                'volume': self._bytes_feature(self.polyp_crop_data.astype(np.float32).tostring()),
                'mask': self._bytes_feature(self.mask.tostring()),
            }))
        writer.write(example.SerializeToString())
        writer.close()



    def colon_mask_dilation(self, iterations=8):
        if self.colon_mask_crop_data == []:
            raise ValueError
        data = ((self.colon_mask_crop_data + self.mask)>0.001).astype(np.uint8)  # Add mask data.
        structure = generate_binary_structure(3,2)
        self.dilated_colon_mask_data = binary_dilation(data, structure, iterations)

#####################################################################################################################
    def read_data_from_raw_file(self):
        if self.raw_vol_dir != 0:
            imageSitk = SimpleITK.ReadImage(self.raw_vol_dir)
            self.polyp_crop_data = SimpleITK.GetArrayFromImage(imageSitk)

        if self.raw_mask_dir != 0:
            imageSitk = SimpleITK.ReadImage(self.raw_mask_dir)
            self.mask = SimpleITK.GetArrayFromImage(imageSitk)

        if self.colon_mask_dir != 0:
            imageSitk = SimpleITK.ReadImage(self.colon_mask_dir)
            self.colon_mask_crop_data = SimpleITK.GetArrayFromImage(imageSitk)

        if self.dilated_colon_mask_dir != 0:
            imageSitk = SimpleITK.ReadImage(self.dilated_colon_mask_dir)
            self.dilated_colon_mask_data = SimpleITK.GetArrayFromImage(imageSitk)


    def write_data_into_raw_file(self):
        if self.raw_vol_dir != 0:
            img = SimpleITK.GetImageFromArray(self.polyp_crop_data)
            SimpleITK.WriteImage(img, self.raw_vol_dir)

        if self.raw_mask_dir != 0:
            img = SimpleITK.GetImageFromArray(self.mask)
            SimpleITK.WriteImage(img, self.raw_mask_dir)

        if self.colon_mask_dir != 0:
            img = SimpleITK.GetImageFromArray(self.colon_mask_crop_data)
            SimpleITK.WriteImage(img, self.colon_mask_dir)

        if self.dilated_colon_mask_dir != 0:
            img = SimpleITK.GetImageFromArray(self.dilated_colon_mask_data)
            SimpleITK.WriteImage(img, self.dilated_colon_mask_dir)

#####################################################################################################################




class Volume_Data:
    def __init__(self):
        self.base_dir = 0
        self.rawdata = 0
        self.colon_mask = 0
        self.shape = 0
    '''Volume'''
    def Set_Directory(self, direc):
        self.base_dir = direc


    def Load_Volume_Data(self):
        if os.path.exists(self.base_dir + "/oriInterpolatedCTData.raw"):
            self.volume_data_dir = self.base_dir + "/oriInterpolatedCTData.raw"
        elif os.path.exists(self.base_dir + "/InterpolatedCTData.raw"):
            self.volume_data_dir = self.base_dir + "/InterpolatedCTData.raw"
        else:
            print("Cannot find CT volume data")
            print(self.base_dir)
            raise IOError

        ds = dicom.read_file(self.volume_data_dir)
        data = ds.pixel_array
        self.rawdata = (data<-999)*-999 + (data>=-999)*data
        self.rawdata = self.rawdata.astype(np.int16)
        self.shape = ds.pixel_array.shape

    def load_colon_mask(self):
        '''Load colon mask from base dir.
        '''
        if self.base_dir == 0:
            print("Base dir not provided!")
            raise IOError
        if self.shape == 0:
            print("Shape not provided!")
            raise ValueError
        if os.path.exists(os.path.join(self.base_dir, "colonMask.raw")):
            self.colon_mask_dir = os.path.join(self.base_dir, "colonMask.raw")
        else:
            print("Cannot find colon mask data.")
            raise IOError
        with open(self.colon_mask_dir, "rb") as f:
            data = f.read()
            data = np.fromstring(data, dtype=np.uint8)
            data = (data>0.00001).astype(np.uint8)    # In case some map of polyp mask have labels of 255.
            self.colon_mask = np.reshape(data, self.shape)

    def colon_mask_dilation(self, iterations=8):
        if self.colon_mask == []:
            raise ValueError
        structure = generate_binary_structure(3,2)
        self.dilated_colon_mask = binary_dilation(self.colon_mask, structure, iterations).astype(np.uint8)

    def load_polyp_mask(self):
        if not os.path.exists(os.path.join(self.base_dir, "polyp_mask.nrrd")):
            return 0
        image = SimpleITK.ReadImage(os.path.join(self.base_dir, "polyp_mask.nrrd"))
        self.polyp_mask = SimpleITK.GetArrayFromImage(image)
        return 1

    def load_score_map(self, name):
        image = SimpleITK.ReadImage(os.path.join(self.base_dir, name))
        return SimpleITK.GetArrayFromImage(image)
