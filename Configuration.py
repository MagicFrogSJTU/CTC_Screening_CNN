# Created by Chen Yizhi, 20171201.

# Directory of the project.
BASE_DIR = '/mnt/disk6/Yizhi/CTC_SCREEN'
# The directory where all CT volume data are stored.
CT_VOLUME_BASE_DIR = "/mnt/disk6/Yizhi/SegmentedColonData"
# Each line of volume_info.txt contains a relative url of a fold of a CT volume data to the CT_VOLUME_BASE_DIR
# A typical line would be:
# TestFile/WRAMC VC-100M/1.3.6.1.4.1.9328.50.99.75748/1.3.6.1.4.1.9328.50.99.75749
CT_VOLUME_DATA_RECORD_FILE = "/mnt/disk6/Yizhi/volume_info.txt"
# The directory where information files would be stored. If you don't know what it is, just put it under the code fold
# of the BASE_DIR
INFORMATION_FILE_DIR = '/mnt/disk6/Yizhi/CTC_SCREEN/code'

# The size for network input.
SCREEN_VOLUME_SIZE = 48
# The size of volume cropped from original CT data.
CUT_RAW_VOLUME_SIZE = 160

################################################################ str in "" is IN REGULAR EXPRESSION!
# The Directory Sturcture of Model:
# BASE_DIR
#           /input
#                   /polypdata
#                               /"\d*", folds for different CT volumes
#                                       /"\d*", folds for polyps
#                                               /image data for a polyp
#           /code(The root directory that you git-clone!)
#                   /".*\.py"
#                   /"version\d", folds for different network versions of code.
#                               /".*\.py"
#                               /"cross_\d", folds for parameters of different cross validation.
#
#################################################################
# The Directory Structure of Inputting CT volume data:
#                       (Generally follow the data structure in database of TCIA and WRAMC)
#
# CT_VOLUME_BASE_DIR
#           /different patient
#                   /".*"
#                           /different volumes of the patient
#                                   /CT_data.nii.gz
#                                   /polyp_mask.nii.gz
#                                   /colon mask.nii.gz


import os

class Configuration:
    cross_validation_url = os.path.join(BASE_DIR, 'input', 'cross_validation')
    independent_fold_url = os.path.join(BASE_DIR, 'input','independent_validation')
    polypdata_fold_url = os.path.join(BASE_DIR, 'input', 'polypdata')
    screening_output_url = os.path.join(BASE_DIR, 'input', 'screen_output')

    checkpoint_fold = os.path.join('Screen','Train')
    eval_fold = os.path.join('Screen', 'Eval')

    cross_index = None

    def cross_validation_model_dir(self,):
        return "cross_" + str(self.cross_index)

    def get_current_model_dir(self):
        pwd = os.getcwd()
        return pwd+"/"+self.cross_validation_model_dir()
    def get_current_checkpoint_dir(self):
        return os.path.join(self.get_current_model_dir(),
                             self.checkpoint_fold)
    def get_current_eval_dir(self):
        return os.path.join(self.get_current_model_dir(),
                            self.eval_fold)




