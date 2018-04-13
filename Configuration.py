# Created by Chen Yizhi, 20171201.
import os

# Directory of the project.
BASE_DIR = '/mnt/disk6/Yizhi/CTC_SCREEN'

# Each line of volume_info.txt contains a relative url of a fold of a CT volume data.
CT_VOLUME_DATA_RECORD_FILE = "/mnt/disk6/Yizhi/volume_info.txt"
# The directory where all CT volume data exist.
CT_VOLUME_BASE_DIR = "/mnt/disk6/Yizhi/SegmentedColonData"
# The directory where information files would be stored
INFORMATION_FILE_DIR = '/mnt/disk6/Yizhi/CTC_SCREEN/code'

# The size for network input.
SCREEN_VOLUME_SIZE = 48
# The size of volume cut from original CT data.
CUT_RAW_VOLUME_SIZE = 160

################################################################ str in "" is IN REGULAR EXPRESSION!
# The Directory Sturcture of Model:
# BASE_DIR
#           /input
#                   /polypdata
#                               /"\d*", folds for different CT volumes
#                                       /"\d*", folds for polyps
#                                               /image data for a polyp
#                   /cross_validation
#                               /"\d", folds for different fold of validation.
#                                       /"testSet.txt"
#                                       /"trainSet.txt"
#                                       /"testVolumeRecord.txt"
#                                       /"trainVolumeRecord.txt"
#           /code
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
#                                   /CT volume data
#                                   /polyp mask
#                                   /colon mask
#                                   /"voxel spacing.txt"


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




