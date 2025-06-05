#
#  This scripts used for generaing a dataset file 
#  using  raw files from IEMOCAP
#  result includes:
#       sample_name, label, start_time, end_time, txt
#  start_time, end_time is for post processping that clips video from 
#  original avi files. 
#


import os
import re

data_path = "/mimer/NOBACKUP/groups/naiss2024-22-578/IEMOCAP_full_release/"

session_ids = ["Session1","Session2","Session3","Session4","Session5" ]


def get_sample_str(txt_path):
    # read txt file and extra sample str
    # eg.: [6.7700 - 8.4600]  Ses01M_impro01_F000  ang  [1.5000, 3.5000, 4.5000]
    sample_str_list = []
    with open(txt_path,'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                sample_str_list.append(line)
    return sample_str_list

def get_sample_info(sample_str: str):
    match = re.search(r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]", sample_str)
    if match:
        start_time, end_time = match.groups()
        # print(f"{start_time}, {end_time}")
        assert float(end_time) - float(start_time) > 0
    else:
        raise BaseException("Cannot get time info:", sample_str)
    
    name_label_match = re.search(r"\[(.*?)\](.*?)\[(.*?)\]", sample_str)
    if name_label_match:
        name_label = name_label_match.group(2).strip()
        itmes = name_label.split("\t")
        name = itmes[0].strip()
        label = itmes[1].strip()
    else:
        raise BaseException("Cannot get name info:", sample_str)
    return start_time, end_time, name, label


def get_scripts(scripts_txt_path):
    sample_name_script_map ={}
    with open(scripts_txt_path, 'r') as f:
        # read script eg. Ses05F_script02_2_M041 [436.5441-438.3820]: Or not.
        # get sample name, and script
        for line in f.readlines():
            line = line.strip()
            if not line.startswith("Ses"):
                continue
            name = line.split(" ")[0]
            txt = line.split(":")[-1].strip()
            if name in sample_name_script_map:
                print("Ignore illegal data:",line)
                continue
            sample_name_script_map[name] = txt
    return sample_name_script_map

            
def collect_sample_str_session(dataset_path, session_id):
    txt_path = "dialog/EmoEvaluation/"
    script_path = "dialog/transcriptions"
    
    print("Handling :", session_id)

    full_path = os.path.join(dataset_path, session_id, txt_path)
    full_script_path = os.path.join(dataset_path, session_id, script_path)
    # read all 
    files =  [ f for f in os.listdir(full_path) \
            if f.endswith(".txt") and f.startswith("Ses")]
        
    print("Get txt files:", len(files))

    sample_str_list = []
    sample_script_map = {}
    for f in files:
        txt_path = os.path.join(full_path,f)
        sample_str_list_i = get_sample_str(txt_path)
        sample_str_list.extend(sample_str_list_i)
        
        script_path = os.path.join(full_script_path,f)
        sample_script_map_i = get_scripts(script_path)
        sample_script_map.update(sample_script_map_i)
        
    
    print("Get sample str:", len(sample_str_list))
    return sample_str_list, sample_script_map

all_sample_str = []
all_sample_script_map = {}
for session_id in session_ids:
    sample_strs_i, sample_script_map_i = collect_sample_str_session(data_path,session_id)
    all_sample_str.extend(sample_strs_i)
    all_sample_script_map.update(sample_script_map_i)


class_count_dict = {}

with open("iemocap_all_sample.txt", "w") as f:
    for sample_str in all_sample_str:
        start_time, end_time, name, label = get_sample_info(sample_str)
        if name not in all_sample_script_map:
            print("skip:", name)
            continue
        f.write("%s, %s, %s, %s ### %s \n" %(name, label, start_time, end_time, all_sample_script_map[name]))

        if label not in class_count_dict:
            class_count_dict[label] = 0
        class_count_dict[label] +=1
        
print("Get total %d samples in 5 sessions" %(len(all_sample_str)))
print("Gen iemocap_all_sample.txt")

labels = list(class_count_dict.keys())
labels.sort()

with open("info.txt", 'w') as f:
    f.write("Class, Number\n")
    for label in labels:
        num = class_count_dict[label]
        f.write("%s : %s\n" %(label, num))       
print("Gen info.txt")