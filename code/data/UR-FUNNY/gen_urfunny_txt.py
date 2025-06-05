# more reference: https://github.com/ROC-HCI/UR-FUNNY/blob/master/README.md


import os
import pickle


file_path = os.path.dirname(os.path.abspath(__file__))

data_root = "/mimer/NOBACKUP/groups/naiss2024-22-578/UR-FUNNY"
txt_pkl = os.path.join(data_root, "language_sdk.pkl")
label_pkl = os.path.join(data_root, "humor_label_sdk.pkl")

# split the data into train, valid, test configuration
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1


# load text data
id_text_map = {}
max_text_len = 0
total_text_len = 0
with open(txt_pkl, 'rb') as f:
    text_data = pickle.load(f)
    ids = text_data.keys()
    for id in ids:
        # we use the punchline sentence for the text data
        # make thing easier
        text = text_data[id]["punchline_sentence"]
        if len(text) > max_text_len:
            max_text_len = len(text)
        total_text_len += len(text)
        if id in id_text_map:
            print("Duplicated id: ", id)
        id_text_map[id] = text
        
print("Total number of text data: ", len(id_text_map))  
print("Max text length: ", max_text_len)
print("Average text length: ", int(total_text_len / len(id_text_map)))
    
# load label data
id_label_map = {}
class_num_statistic = {}
with open(label_pkl, 'rb') as f:
    label_data = pickle.load(f)
    ids = label_data.keys()
    print("Number of label data: ", len(ids))
    for id in ids:
        label = label_data[id]
        if id in id_label_map:
            print("Duplicated id: ", id)
        id_label_map[id] = label
        if label in class_num_statistic:
            class_num_statistic[label] += 1
        else:
            class_num_statistic[label] = 1
    
print("Total number of label data: ", len(id_label_map))
print("Label statistic: ", class_num_statistic)
print("Label: 0: not funny, 1: funny")

# save the text and label data
save_path = os.path.join(file_path, "ur_funny_all_text_label.txt")
# save in the format: id, label ### text
with open(save_path, 'w') as f:
    for id in id_text_map:
        text = id_text_map[id]
        label = id_label_map[id]
        f.write("{}, {} ### {}\n".format(id, label, text))
        
print("Save all text and label data to: ", save_path)

# split the data into train, valid, test
train_path = os.path.join(file_path, "ur_funny_train.txt")
valid_path = os.path.join(file_path, "ur_funny_valid.txt")
test_path = os.path.join(file_path, "ur_funny_test.txt")

train_num = int(len(id_text_map) * train_ratio)
valid_num = int(len(id_text_map) * valid_ratio)
test_num = len(id_text_map) - train_num - valid_num


keys = id_text_map.keys()

keys = list(keys)
keys.sort()
# shuffle the data
import random
random.seed(0)
random.shuffle(keys)

train_ids = keys[:train_num]
valid_ids = keys[train_num:train_num+valid_num]
test_ids = keys[train_num+valid_num:]


print("Train: ", len(train_ids))
print("Valid: ", len(valid_ids))
print("Test: ", len(test_ids))

# save train data
def save_data(ids, path):
    with open (path, 'w') as f:
        for id in ids:
            text = id_text_map[id]
            label = "NOTFUNNY" if id_label_map[id] == 0 else "FUNNY"
            f.write("{}, {}, ### {}\n".format(id, label, text))
            
save_data(train_ids, train_path)
print("Save train data to: ", train_path)
save_data(valid_ids, valid_path)
print("Save valid data to: ", valid_path)
save_data(test_ids, test_path)
print("Save test data to: ", test_path)

# gen stats file
train_statistic = {}
valid_statistic = {}
test_statistic = {}

def gen_statistic(ids, statistic):
    for id in ids:
        label = id_label_map[id]
        if label in statistic:
            statistic[label] += 1
        else:
            statistic[label] = 1

stats_path = os.path.join(file_path, "ur_funny_info.txt")
with open(stats_path, 'w') as f:
    f.write("Total number of text data: {}\n".format(len(id_text_map)))
    f.write("Total number of label data: {}\n".format(len(id_label_map)))
    f.write("Label statistic: {}\n".format(class_num_statistic))
    
    f.write("Train: {}\n".format(len(train_ids)))
    gen_statistic(train_ids, train_statistic)
    f.write("Train statistic: {}\n".format(train_statistic))
    
    f.write("Valid: {}\n".format(len(valid_ids)))
    gen_statistic(valid_ids, valid_statistic)
    f.write("Valid statistic: {}\n".format(valid_statistic))
    
    f.write("Test: {}\n".format(len(test_ids)))
    gen_statistic(test_ids, test_statistic)
    f.write("Test statistic: {}\n".format(test_statistic))
    
    f.write("Max text length: {}\n".format(max_text_len))
    f.write("Average text length: {}\n".format(int(total_text_len / len(id_text_map))))

with open(os.path.join(file_path, "ur_funny_stat.txt"), 'w') as f:
    f.write("NOTFUNNY\n")
    f.write("FUNNY")

print("Done!")