from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

from shutil import copyfile

def readFile(csv_file_path):
    csv_contents = list()
    csv_keys = list()
    csv_values = list()
    f = open(csv_file_path, 'rt')
    try:
        reader = csv.reader(f)
        for row in reader:
            csv_keys.append(row[0])
            csv_values.append(row[1])
            csv_contents.append(row)
        csv_contents.pop(0)
        csv_keys.pop(0)
        csv_values.pop(0)
        unique_values = list(set(csv_values))
    finally:
        f.close()
    return csv_contents, csv_keys, csv_values, unique_values

def createDirs(dir, unique_values):
    for value in unique_values:
        if not os.path.exists(dir + value):
            os.makedirs(dir + value)

def copyFiles(png_file_path, destination, csv_contents):
    for record in csv_contents:
        print('current record: ' + str(record))
        source = png_file_path + record[0] + '.png'
        dest = destination + record[1] + '/' + record[0] + '.png'

        print('copy source: ' + source)
        print('copy destination: ' + dest)
        copyfile(source, dest)

csv_train_file_path = '/home/user/Dropbox/college/4th year/FYP/datasets/cifar-10/trainLabels.csv'
train_png_file_path = '/home/user/Dropbox/college/4th year/FYP/datasets/cifar-10/train/'

#get all labeled data from training data
csv_contents, csv_keys, csv_values, unique_values = readFile(csv_train_file_path)

num_train = int(round((len(csv_keys) / 100.) * 80))
num_test = len(csv_keys) - num_train

# divide labeled training data into labeled training and test
csv_contents_train = csv_contents[0:num_train]
csv_contents_test = csv_contents[num_train:]
csv_keys_train = csv_keys[0:num_train]
csv_keys_test = csv_keys[num_train:]
csv_values_train = csv_values[0:num_train]
csv_values_test = csv_values[num_train:]

# copy files into:
#  - test/<unique_label>
#  - train/<unique_label>
processed_data_trn = 'pro-data/train/'
createDirs(processed_data_trn, unique_values)
copyFiles(train_png_file_path, processed_data_trn, csv_contents_train)

processed_data_tst = 'pro-data/test/'
createDirs(processed_data_tst, unique_values)
copyFiles(train_png_file_path, processed_data_tst, csv_contents_test)
