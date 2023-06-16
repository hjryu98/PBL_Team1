import os
import shutil
import glob

speakers = ['p262', 'p272', 'p229', 'p232', 'p292', 'p293', 'p360', 'p361', 'p248', 'p251']

train_destination = './data/mc/train/'
test_destination = './data/mc/test/'

train_files = glob.glob('./data/mc/train/*/*')
test_files = glob.glob('./data/mc/test/*/*')
# print(len(train_folders))

# move .npy files
for train_file_path in train_files:
    shutil.move(train_file_path, train_destination)
for test_file_path in test_files:
    shutil.move(test_file_path, test_destination)

# delete remaining folders
for train_speaker in speakers:
    train_speaker_folder = os.path.join(train_destination, train_speaker)
    shutil.rmtree(train_speaker_folder)
for test_speaker in speakers:
    test_speaker_folder = os.path.join(test_destination, test_speaker)
    shutil.rmtree(test_speaker_folder)