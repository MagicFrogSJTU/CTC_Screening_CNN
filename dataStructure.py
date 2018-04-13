import numpy as np
import dicom
import tensorflow as tf
import os
import SimpleITK
import time
from scipy.ndimage import binary_dilation
from scipy.ndimage import generate_binary_structure

class Polyp_data:
    '''data object for polyp'''
    raw_CT_name = "CT.nii.gz"
    raw_mask_name = "mask.nii.gz"
    colon_mask_name = "colon_mask.nii.gz"
    dilated_colon_mask_name = "dilated_colon_mask.nii.gz"
    raw_tf_name = "data.tf"                              # vol: float32, mask: uint8


    def __init__(self):
        # directory would be accessed when read_data_from_raw_file() or write
        self.base_dir = 0

        self.raw_CT_dir = 0
        self.raw_mask_dir = 0
        self.colon_mask_dir = 0
        self.dilated_colon_mask_dir = 0

        # variables containing 3D data.
        self.crop_CT_data = 0             # int16
        self.crop_polyp_mask = 0                        # uint8
        self.crop_colon_mask = 0

    def calculate_propety(self, polyp_mask, polyp_label):
        self.label_in_polypmask = polyp_label
        mask = polyp_mask == polyp_label
        dots = np.where(mask!=0)
        self.pixelsize_of_polyp = len(dots[0])
        self.coord_of_center = [np.average(nums) for nums in dots]
        assert len(self.coord_of_center)==3
        self.coord_of_center = np.array(self.coord_of_center)

    def Crop_Polyp_CTData(self, volume_data, crop_size, coord_of_center):
        '''size must be a int. homogeneous'''
        # Compute center point of dots
        assert type(crop_size) == int
        assert crop_size%2 == 0
        CT_data = volume_data.CT_data
        colon_mask = volume_data.colon_mask
        polyp_mask = volume_data.polyp_mask

        self.crop_CT_data = -999 * np.ones((crop_size,crop_size,crop_size),dtype=np.int16)
        self.crop_colon_mask = np.zeros((crop_size,crop_size,crop_size), dtype=np.uint8)
        self.crop_polyp_mask = np.zeros((crop_size,crop_size,crop_size), dtype=np.uint8)

        #print "aver: ", aver
        copy_ini = coord_of_center- crop_size/2
        copy_end = coord_of_center + crop_size/2
        paste_ini = [0,0,0]
        paste_end = [crop_size, crop_size, crop_size]
        for i in range(0,3):
            if copy_ini[i] < 0:
                paste_ini[i] = 0 - copy_ini[i]
                copy_ini[i] = 0
                pass
            elif copy_end[i] > CT_data.shape[i]:
                paste_end[i] = crop_size - (copy_end[i]-CT_data.shape[i])
                copy_end[i] = CT_data.shape[i]

        #dot_ini = copy_end.astype(np.int)
        #dot_end = copy_ini.astype(np.int)
        self.crop_CT_data[paste_ini[0]:paste_end[0],
                             paste_ini[1]:paste_end[1],
                             paste_ini[2]:paste_end[2]] = \
            CT_data[copy_ini[0]:copy_end[0],
                        copy_ini[1]:copy_end[1],
                        copy_ini[2]:copy_end[2]]

        self.crop_colon_mask[paste_ini[0]:paste_end[0],
                             paste_ini[1]:paste_end[1],
                             paste_ini[2]:paste_end[2]] = \
            colon_mask[copy_ini[0]:copy_end[0],
                        copy_ini[1]:copy_end[1],
                        copy_ini[2]:copy_end[2]]

        # Generate mask, which corresponds to the crop data.
        self.crop_polyp_mask[paste_ini[0]:paste_end[0],
                  paste_ini[1]:paste_end[1],
                  paste_ini[2]:paste_end[2]] = \
             polyp_mask[copy_ini[0]:copy_end[0],
                  copy_ini[1]:copy_end[1],
                  copy_ini[2]:copy_end[2]]

    def _bytes_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def save_data_into_tf_file(self):
        assert self.base_dir != 0
        writer = tf.python_io.TFRecordWriter(os.path.join(self.base_dir, self.raw_tf_name))
        example = tf.train.Example(features=tf.train.Features(feature={
                'volume': self._bytes_feature(self.crop_CT_data.astype(np.int16).tostring()),
                'polyp_mask': self._bytes_feature(self.crop_polyp_mask.astype(np.uint8).tostring()),
                'colon_mask': self._bytes_feature(self.crop_colon_mask.astype(np.uint8).tostring()),
            }))
        writer.write(example.SerializeToString())
        writer.close()



    def colon_mask_dilation(self, iterations=8):
        if self.crop_colon_mask == []:
            raise ValueError
        data = ((self.crop_colon_mask + self.crop_polyp_mask)!=0).astype(np.uint8)  # Add mask data.
        structure = generate_binary_structure(3,2)
        self.dilated_crop_colon_mask = binary_dilation(data, structure, iterations)

#####################################################################################################################
    def read_data_from_raw_file(self):
        if self.raw_CT_dir != 0:
            imageSitk = SimpleITK.ReadImage(self.raw_CT_dir)
            self.crop_CT_data = SimpleITK.GetArrayFromImage(imageSitk)

        if self.raw_mask_dir != 0:
            imageSitk = SimpleITK.ReadImage(self.raw_mask_dir)
            self.crop_polyp_mask = SimpleITK.GetArrayFromImage(imageSitk)

        if self.colon_mask_dir != 0:
            imageSitk = SimpleITK.ReadImage(self.colon_mask_dir)
            self.crop_colon_mask = SimpleITK.GetArrayFromImage(imageSitk)


    def write_data_into_raw_file(self):
        '''Set one or more of the dirs, and save the corresponding data to the disk,
            '''
        self.raw_CT_dir = os.path.join(self.base_dir, self.raw_CT_name)
        self.raw_mask_dir = os.path.join(self.base_dir, self.raw_mask_name)
        self.colon_mask_dir = os.path.join(self.base_dir, self.colon_mask_name)
        if self.raw_CT_dir != 0:
            img = SimpleITK.GetImageFromArray(self.crop_CT_data)
            SimpleITK.WriteImage(img, self.raw_CT_dir)

        if self.raw_mask_dir != 0:
            img = SimpleITK.GetImageFromArray(self.crop_polyp_mask)
            SimpleITK.WriteImage(img, self.raw_mask_dir)

        if self.colon_mask_dir != 0:
            img = SimpleITK.GetImageFromArray(self.crop_colon_mask)
            SimpleITK.WriteImage(img, self.colon_mask_dir)


#####################################################################################################################




class Volume_Data:
    '''Structure containing all data for a CT Volume.
        Assume all data of a volume are in the same fold, as self.base_dir
        Here My Data is in all kinds of strange formats, Thus
        you should implement your own LOAD functions.'''
    #TODO: Make it a base class. I\O functions should be implemented in another class.
    def __init__(self):
        self.base_dir = 0

        self.patient_uid = None
        self.volume_uid = None

        self.shape = 0

        self.CT_data = 0
        self.colon_mask = 0
        self.polyp_mask = 0
        self.dilated_colon_mask = 0
        self.score_map = 0

    def Set_Directory(self, direc):
        self.base_dir = direc
        while direc[len(direc)-1] == '/':
            direc = direc[:len(direc)-1]
        # Delete two floors. As it may occur in some situations where only by this could it be reasonable.
        tmp = direc[:direc.rfind("/")]
        self.patient_dir = tmp[:tmp.rfind("/")]
        self.patient_uid = self.patient_dir[self.patient_dir.rfind("/")+1:]
        self.volume_uid = direc[len(self.patient_dir)+1:]

    def clear_volume_data(self):
        self.CT_data = 0
        self.colon_mask = 0
        self.dilated_colon_mask = 0
        self.polyp_mask = 0
        self.score_map = 0


    def load_volume_data(self):
        '''Load CT volume data.
        '''
        self.volume_data_dir = 0
        names=['oriInterpolatedCTData.raw', 'InterpolatedCTData.raw']
        for name in names:
            if os.path.exists(os.path.join(self.base_dir, name)):
                self.volume_data_dir = os.path.join(self.base_dir, name)
                break
        assert self.volume_data_dir != 0

        ds = dicom.read_file(self.volume_data_dir)
        data = ds.pixel_array
        self.CT_data = (data<-999)*-999 + (data>=-999)*data
        self.CT_data = self.CT_data.astype(np.int16)
        self.shape = ds.pixel_array.shape

    def load_colon_mask(self):
        '''Load colon mask from base dir.
        '''
        name = 'colonMask.raw'
        if self.base_dir == 0:
            print("Base dir not provided!")
            raise IOError
        if self.shape == 0:
            print("Shape not provided!")
            raise ValueError
        if os.path.exists(os.path.join(self.base_dir, name)):
            self.colon_mask_dir = os.path.join(self.base_dir, name)
        else:
            print("Cannot find colon mask data.")
            raise IOError
        with open(self.colon_mask_dir, "rb") as f:
            data = f.read()
            data = np.fromstring(data, dtype=np.uint8)
            data = (data!=0).astype(np.uint8)    # In case some map of polyp mask may have labels of 255.
            self.colon_mask = np.reshape(data, self.shape)

    def colon_mask_dilation(self, iterations=8):
        if self.colon_mask is 0:
            raise ValueError
        structure = generate_binary_structure(3,2)
        self.dilated_colon_mask = binary_dilation(self.colon_mask, structure, iterations).astype(np.uint8)

    def load_polyp_mask(self):
        name = 'polyp_mask.nrrd'
        if not os.path.exists(os.path.join(self.base_dir, name)):
            return 0
        image = SimpleITK.ReadImage(os.path.join(self.base_dir, name))
        self.polyp_mask = SimpleITK.GetArrayFromImage(image)
        return 1

    def load_score_map(self, fold=0):
        '''For nested cross validation, score maps are stored in the cross_1, cross_2, cross3... directory
        Under the volume unit direcotry.
        '''
        name='score_map.nii.gz'
        base_path = os.path.join(self.base_dir, fold, name)
        if not os.path.exists(base_path):
            return 0
        image = SimpleITK.ReadImage(base_path)
        self.score_map = SimpleITK.GetArrayFromImage(image)
        return self.score_map

    def load_colon_dilation(self):
        name = "dilated_colon_mask.nii.gz"
        if not os.path.exists(os.path.join(self.base_dir, name)):
            return 0
        image = SimpleITK.ReadImage(os.path.join(self.base_dir, name))
        self.dilated_colon_mask = SimpleITK.GetArrayFromImage(image)
        return 1

    def load_spacing(self):
        name = "voxelSpacing.txt"
        with open(os.path.join(self.base_dir, name), 'r') as f:
            nums = f.read().split()
            floats = [0,0,0]
            for i in range(3):
                floats[i] = float(nums[i])
            self.spacing = floats
