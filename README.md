Tensorflow implementation of "A 3D Convolutional Neural Network Framework for Polyp Candidates Detection
with Limited Dataset in CT Colonography", Chen Yizhi, 2018, EMBC. Copyright Reserved. Free for all kinds of reproduction and revisement for research purposes.
Under Tensorflow1.4, Python2.7, Ubuntu16.04

# FILE STRUCTURE
You should refer to Configuration.py for a full understanding of the file structure of the program and the database.

# DATA INPUT
In order to avoid the considerable time needed to load full-size CT volumes when training, we would first crop the
volume and then organized them as a separate POLYP DATASET.
1. Prepare you CT colonography data as in the Configuration.py.
2. List the directories of all the CT volumes in a text file.
2. Modify several important directory variables in the Configuation.py.
3. Run dataBase.py to construct information files and the polyp dataset.

# Training
"cd version2" and "python train.py --fold=0" to train your model for the first cross-validation fold!

# Validation and Testing
1. "cd version2" and "python eval.py --fold=0" for a validation in every 120 seconds.
2. "cd version2" and "python screen.py --fold=0" for a test to the full-size CT volumes of the test set.

Star me if you like it!



