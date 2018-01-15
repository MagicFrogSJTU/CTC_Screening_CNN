import sys
import os

sys.path.append(os.getcwd() + "/..")
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import screen_volume
from polyp_def import Volume_Data

volume = Volume_Data()
#volume.base_dir = ("/mnt/disk6/Yizhi/SegmentedColonData/TestFile2/WRAMC VC-403M/1.3.6.1.4.1.9328.50.99.338779/1.3.6.1.4.1.9328.50.99.338780")
#volume.base_dir = ("/mnt/disk6/Yizhi/SegmentedColonData/TrainFile/SD VC-502M/1.3.6.1.4.1.9328.50.6.308416/1.3.6.1.4.1.9328.50.6.308417")
#volume.base_dir = ("/mnt/disk6/Yizhi/SegmentedColonData/CTC-1639466381/1.3.6.1.4.1.9328.50.81.73199297624683292865461941105982569072/1.3.6.1.4.1.9328.50.81.249772276464185657199294455739635725693")
#volume.base_dir = ("/mnt/disk6/Yizhi/SegmentedColonData/TestFile2/WRAMC VC-226M/1.3.6.1.4.1.9328.50.99.188876/1.3.6.1.4.1.9328.50.99.188877")
volume.base_dir = ("/mnt/disk6/Yizhi/SegmentedColonData/1.3.6.1.4.1.9328.50.4.0080/1.3.6.1.4.1.9328.50.4.86204/1.3.6.1.4.1.9328.50.4.86213")
volume.load_score_map()
volume.Load_Volume_Data()
volume.load_polyp_mask()

screen_volume.segmentation(volume, 0.99, 0.9)
num_gold, num_correct , num_false = screen_volume.confirm(volume)
print(num_gold, num_correct, num_false)