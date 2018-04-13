''' Created by Yizhi Chen. 20170930'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os.path
import numpy as np
import os
import dataStructure
import Configuration
import shutil
import pandas as pd
from Configuration import INFORMATION_FILE_DIR
import SimpleITK as sitk





class DataBase:
    def __init__(self, whichfold=None):
        self.df = None
        if whichfold != None:
            self.load_database_info_from_record_file()
            self.df['fold'] = self.df['fold%s'%whichfold].map(lambda x:x)

    def init_first_time(self):
        '''Set up all kinds of information collecting and database building for CTC Screening'''
        self.set_up_from_volume_record()
        self.build_polyp_info()
        self.build_cross_validation_split_randomly()
        self.df.to_csv(os.path.join(INFORMATION_FILE_DIR, 'database_info.csv'),
                       index=False)
        self.df.to_excel(os.path.join(INFORMATION_FILE_DIR, 'database_info.xls'),
                         index=False)
        config = Configuration.Configuration()
        self.set_up_screen_database(config.polypdata_fold_url)


    def load_database_info_from_record_file(self, file="database_info.csv"):
        '''load the information of database from a record file from disk.
        Args:
            file: the path of file, which can be csv or xls file.
            '''
        file = os.path.join(INFORMATION_FILE_DIR, file)
        file_type = file[file.rfind('.')+1:]
        if not os.path.exists(file):
            print("You should build the information file first!")
            raise IOError
        if file_type == 'csv':
            self.df = pd.read_csv(file)
        elif file_type == 'xls':
            self.df = pd.read_excel(file)
        else:
            raise NotImplementedError
        def func(coord_str):
            coord_str = coord_str[1:-1]
            coord_str = coord_str.split()
            return np.array([int(i) for i in coord_str])
        self.df['coord of center'] = self.df['coord of center'].map(func)

    def save_database_info(self, file_prefix="database_info"):
        direc = os.path.join(INFORMATION_FILE_DIR, file_prefix)
        self.df.to_csv(direc+".csv")
        self.df.to_excel(direc+".xls")

    def set_up_from_volume_record(self, record_url=0, base_url=0):
        if record_url is 0:
            record_url = Configuration.CT_VOLUME_DATA_RECORD_FILE
        if base_url is 0:
            base_url = Configuration.CT_VOLUME_BASE_DIR

        with open(record_url, 'r') as f:
            lines = f.readlines()
        volume_list = []
        for volume_index, line in enumerate(lines):
            line = line[:-1]
            line = os.path.join(base_url, line)
            new_volume = dataStructure.Volume_Data()
            new_volume.Set_Directory(line)
            new_unit = {'volume path': line,
                        'volume uid':new_volume.volume_uid,
                        'patient uid': new_volume.patient_uid,
                        }
            for file in os.listdir(line):
                if file == 'oriInterpolatedCTData.raw' or file == 'InterpolatedCTData.raw':
                    volume_list.append(new_unit)
                    break
        self.df = pd.DataFrame(volume_list)

    def build_polyp_info(self):
        new_rows = []
        polyp_index = 0
        for index, row in self.df.iterrows():
            print('\r',index, end='')
            volume_data = dataStructure.Volume_Data()
            volume_data.Set_Directory(row['volume path'])
            volume_data.load_spacing()
            row['spacing'] = volume_data.spacing
            if not volume_data.load_polyp_mask():
                new_row = row.copy()
                new_row['has polyp'] = False
                new_rows.append(new_row)
            else:
                label_max = np.max(volume_data.polyp_mask)
                assert label_max >= 1
                for label in range(1, label_max+1):
                    mask = volume_data.polyp_mask == label
                    dots = np.where(mask!=0)
                    pixelsize_of_polyp = len(dots[0])
                    assert pixelsize_of_polyp > 0
                    coord_of_center = [np.average(nums) for nums in dots]
                    assert len(coord_of_center)==3
                    coord_of_center = np.array(coord_of_center, dtype=np.int32)
                    new_row = row.copy()
                    new_row['has polyp'] = True
                    new_row['pixel size of polyp'] = pixelsize_of_polyp
                    new_row['coord of center'] = coord_of_center
                    new_row['label in polyp mask'] = label
                    new_row['polyp index'] = polyp_index
                    polyp_index += 1
                    new_rows.append(new_row)
            volume_data.clear_volume_data()
        self.df = pd.DataFrame(new_rows)
        print("Number of polyps for all volumes:", polyp_index)

    def set_up_screen_database(self, target_url):
        '''Crop the CT image into little cube centered on polyps. Set up directories on the target url.
            Polyps are organized in seperate folds, whose name is the polyp index.
        Args:
            target_url: the root url, where the polyp database would be set up.
            '''
        assert self.df is not None

        if os.path.exists(target_url):
            shutil.rmtree(target_url)
        os.makedirs(target_url)

        polyp_num = self.df['polyp index'].max()
        i = 0
        for n_group, n_rows in self.df.groupby('volume path'):
            volume_path = n_group
            volume_data = dataStructure.Volume_Data()
            volume_data.Set_Directory(volume_path)
            volume_data.load_volume_data()
            volume_data.load_polyp_mask()
            if not volume_data.load_colon_dilation():
                raise NotImplementedError
            volume_data.colon_mask = volume_data.dilated_colon_mask

            for index, row in n_rows.iterrows():
                polyp_index = row['polyp index']
                assert polyp_index!=None
                fold_dir = os.path.join(target_url, str(polyp_index))
                os.mkdir(fold_dir)
                polyp_data = dataStructure.Polyp_data()
                polyp_data.base_dir = fold_dir
                polyp_data.Crop_Polyp_CTData(volume_data, Configuration.CUT_RAW_VOLUME_SIZE, row['coord of center'])
                polyp_data.write_data_into_raw_file()
                polyp_data.save_data_into_tf_file()
                print("\r%i/%i" %(i, polyp_num), end='')
                i += 1
            volume_data.clear_volume_data()

    def build_cross_validation_split_randomly(self, folds_num=5, nested_vali_ratio=0.1):
        '''Implement nested cross validation.
            Randomly set a cross validation randomly, with respect to patients.
        Args:
            folds_num: The number of folds for cross validation.
            nested_vali_ratio: The validation dataset ratio in the splited training dataset.
            '''
        assert self.df is not None

        patient_uids = set(self.df['patient uid'])
        num = len(patient_uids)
        print("Number of patients:", num)
        patient_list = list(patient_uids)
        np.random.shuffle(patient_list)
        chunk = num//folds_num+1

        for i in range(folds_num):
            testSet = set(patient_list[chunk*i:chunk*(i+1)])
            trainSet_and_validationSet = patient_uids - testSet
            len_vali = int(len(trainSet_and_validationSet)*nested_vali_ratio)
            assert len_vali > 5
            tmp = list(trainSet_and_validationSet)
            np.random.shuffle(tmp)
            trainSet = set(tmp[len_vali:])
            valiSet = set(tmp[:len_vali])

            new_rows = []
            for index, row in self.df.iterrows():
                if row['patient uid'] in trainSet:
                    row['fold%s'%i] = 'train'
                elif row['patient uid'] in testSet:
                    row['fold%s'%i] = 'test'
                elif row['patient uid'] in valiSet:
                    row['fold%s'%i] = 'validation'
                else:
                    raise ValueError
                new_rows.append(row)
            self.df = pd.DataFrame(new_rows)

        print("!")














if __name__ == '__main__':
    if 0:
        database = DataBase()
        database.init_first_time()
    if 0:
        database = DataBase(0)
        print("!!")
    if 1:
        db = DataBase(0)
        for volume_path, n_group in db.df.groupby('volume path'):
            vdata = dataStructure.Volume_Data()
            vdata.Set_Directory(volume_path)
            vdata.load_volume_data()
            vdata.load_polyp_mask()
            vdata.load_colon_dilation()
            vdata.load_colon_mask()

            pairs = {'CT_data.nii.gz': vdata.CT_data,
                     'polyp_mask.nii.gz': vdata.polyp_mask,
                     'colon_mask.nii.gz': vdata.colon_mask,
                }
            for name in pairs:
                image = sitk.GetImageFromArray(pairs[name])
                sitk.WriteImage(image, os.path.join(volume_path, name))
            print(volume_path)
