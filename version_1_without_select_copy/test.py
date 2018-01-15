import os

for path, folds, files in os.walk('/mnt/disk6/Yizhi/polypdata'):
    for file in files:
        if file == 'CT_volume_base_dir.txt':
            with open(os.path.join(path,file), 'r') as f:
                line = f.read()
            if line.find('/mnt/disk6/Yizhi/SegmentedColonData/TrainFile2/SD VC-419M') != -1:
                print(path)
                print('got')
