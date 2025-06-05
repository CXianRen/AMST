# split iemo dataset into train, test, valid
# due to the imbalance of the dataset, 
# we need to split the dataset by the label
# more refer: Doc/dataset/IEMOCAP.md

# we will save
# ang : 1103
# exc : 1041
# fru : 1849
# neu : 1708
# sad : 1084

import os
import random

class_list = ['ang', 'exc', 'fru', 'neu', 'sad']

class_dict = {}

split_ratio = [0.8, 0.1, 0.1]
seed = 0
random.seed(seed)

# input file
iemocap_all_sample_path = './iemocap_all_sample.txt'

# output files
stat_file = './stat_iemocap.txt'
info_file = './split_info.txt'
train_file = './iemocap_train.txt'
test_file = './iemocap_test.txt'
valid_file = './iemocap_valid.txt'



# read all sample from the dataset
# eg. Ses01M_impro01_F000, ang, 6.7700, 8.4600 ### Next. 
with open(iemocap_all_sample_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line == '': continue
        items = line.split(',')
        if len(items) < 2: 
            print('error line:', line)
            continue
        sample_name = items[0]
        label = items[1].strip()
        if label not in class_list:
            # skip the sample that not in the class_list
            continue
        if label not in class_dict:
            class_dict[label] = []
        class_dict[label].append(line)

# split the dataset
train_class_dict = {}
test_class_dict = {}
valid_class_dict = {}

for label in class_list:
    samples = class_dict[label]
    random.shuffle(samples)
    num_samples = len(samples)
    num_train = int(num_samples * split_ratio[0])
    num_test = int(num_samples * split_ratio[1])
    num_valid = num_samples - num_train - num_test
    train_class_dict[label] = samples[:num_train]
    test_class_dict[label] = samples[num_train:num_train+num_test]
    valid_class_dict[label] = samples[num_train+num_test:]

# save the split dataset
def save_split_dataset(file, class_dict):
    all_samples = []
    for label in class_list:
        all_samples += class_dict[label]
    random.shuffle(all_samples)

    with open(file, 'w') as f:
        for sample in all_samples:
            f.write(sample + '\n')

print('save split dataset...')
save_split_dataset(train_file, train_class_dict)
save_split_dataset(test_file, test_class_dict)
save_split_dataset(valid_file, valid_class_dict)

print('save stat file...')
# save stat_ieomcap.txt
with open(stat_file, 'w') as f:
    class_list.sort()
    for label in class_list:
        if label != class_list[-1]:
            f.write("%s\n" % label)
        else:
            f.write("%s" % label)

print('save split_info.txt...')
# save split_info.txt
with open(info_file, 'w') as f:
    f.write('split_ratio: %s\n' % split_ratio)
    f.write('seed: %s\n' % seed)
    # num of samples in each class
    f.write('num of samples in each class:\n')
    f.write('class: train, test, valid\n')
    for label in class_list:
        num_train = len(train_class_dict[label])
        num_test = len(test_class_dict[label])
        num_valid = len(valid_class_dict[label])
        f.write('%s: %d, %d, %d\n' % (label, num_train, num_test, num_valid))
    
    # num of samples in each split
    f.write('num of samples in each split:\n')
    total_train = sum([len(train_class_dict[label]) for label in class_list])
    f.write('train: %d\n' % total_train)
    total_test = sum([len(test_class_dict[label]) for label in class_list])
    f.write('test: %d\n' % total_test)
    total_valid = sum([len(valid_class_dict[label]) for label in class_list])
    f.write('valid: %d\n' % total_valid)
