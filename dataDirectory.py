# Created by Chen Yizhi, 20171201.
import os


class DataDirectory:
    base_dir = '/mnt/disk6/Yizhi/'
    checkpoint_fold = 'Screen/Train'
    eval_fold = 'Screen/Eval'
    cross_validation_fold = 'cross_validation'
    independent_fold = 'independent_validation'

    def data_base_dir(self):
        return self.base_dir + 'polypdata/'

    def cross_validation_item(self, num=0):
        return "cross_" + str(num)
    def cross_validation_dir(self, num=0):
        return self.base_dir + self.cross_validation_fold + '/' + self.cross_validation_item(num) + '/'

    def independent_validation_item(self, num=1):
        return "independent_" + str(num)
    def independent_dir(self, num=1):
        return self.base_dir + self.independent_fold + "/" + self.independent_validation_item(num) + '/'

    def raw_CTdata_dir(self):
        return self.base_dir + 'SegmentedColonData/'


    # !! User decides!
    def get_current_model_dir(self):
        pwd = os.getcwd()
        return pwd+"/"+self.independent_validation_item()
    def get_current_checkpoint_dir(self):
        return os.path.join(self.get_current_model_dir(),
                             self.checkpoint_fold)
    def get_current_record_dir(self):
        return self.independent_dir()
    def get_current_test_record_dir(self):
        return self.get_current_record_dir()+"/testSet.txt"
    def get_current_train_record_dir(self):
        return self.get_current_record_dir()+"/trainSet.txt"




