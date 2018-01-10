import os

for path, folds, files in os.walk('/mnt/disk6/Yizhi/polypdata'):
    for file in files:
        if file == 'CT_volume_base_dir.txt':
            with open(os.path.join(path,file), 'r') as f:
                line = f.read()
            if line.find('/mnt/disk6/Yizhi/SegmentedColonData/TrainFile2/SD VC-279M/1.3.6.1.4.1.9328.50.6.138038/1.3.6.1.4.1.9328.50.6.1386') != -1:
                print(path)
                print('got')
