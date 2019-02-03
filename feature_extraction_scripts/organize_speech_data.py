#should help w Windows, Macs, Linux operating systems
from pathlib import Path, PurePath
#saving labels
import csv
import random
from feature_extraction_scripts.errors import DatasetMixing

def collect_labels(data_path):
    p = Path(data_path)
    labels = list(p.glob('*/'))
    labels = [PurePath(labels[i]) for i in range(len(labels))]
    labels = [x.parts[1] for x in labels if '_' not in x.parts[1]]
    labels = check_4_github_files(labels)

    return labels

def check_4_github_files(labels_list):
    if 'README.md' in labels_list:
        labels_list.remove('README.md')
    if 'LICENSE' in labels_list:
        labels_list.remove('LICENSE')
    return labels_list

def collect_audio_and_labels(data_path):
    '''
    expects wavefiles to be in subdirectory: 'data'
    labels are expected to be the names of each subdirectory in 'data'
    speaker ids are expected to be the first section of each wavefile
    '''
    p = Path(data_path)
    waves = list(p.glob('**/*.wav'))
    #remove directories with "_" at the beginning
    paths = [PurePath(waves[i]) for i in range(len(waves)) if waves[i].parts[1][0]!="_"]
    labels = [j.parts[1] for j in paths ]
    
    return paths, labels

def create_save_dict_labels_encode(labels_class,head_folder):
    labels_sorted = sorted(labels_class)
    dict_labels_encoded = {}
    for i, label in enumerate(labels_sorted):
        dict_labels_encoded[label] = i
    save_class_labels(labels_sorted,head_folder)
    return dict_labels_encoded

def save_class_labels(sorted_labels,head_folder):
    dict_labels = {}
    for i, label in enumerate(sorted_labels):
        dict_labels[i] = label
    filename = '{}/labels_encoded.csv'.format(head_folder)
    with open(filename,'w') as f:
        w = csv.writer(f)
        w.writerows(dict_labels.items())
    
    return None

def log_class4balance(max_num_per_class,class_max_samps,head_folder):
    filename = '{}/features_log.csv'.format(head_folder)
    with open(filename, 'a', newline='') as f:
        w = csv.writer(f)
        info = {"class fewest samples":class_max_samps,"max num samps per class":max_num_per_class}
        w.writerows(info.items())
        
    return None

def log_extraction_settings(dict_extraction_settings,head_folder):
    filename = '{}/features_log.csv'.format(head_folder)
    with open(filename, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerows(dict_extraction_settings.items())
    
    return None

def get_class_distribution(labels_class,labels_list): 
    dict_class_distribution = {}
    for label in labels_class:
        count = 0
        for label_item in labels_list:
            if label == label_item:
                count+=1
            dict_class_distribution[label] = count
    
    return dict_class_distribution
        

def get_max_samples_per_class(labels_class, labels_list):
    dict_class_distribution = get_class_distribution(labels_class,labels_list)
    max_val = (1000000, None)
    for key, value in dict_class_distribution.items():
        if value < max_val[0]:
            max_val = (value, key)
    
    return max_val


def get_max_nums_train_val_test(max_num_per_class):
    max_train = int(max_num_per_class*.8)
    max_val = int(max_num_per_class*.1)
    max_test = int(max_num_per_class*.1)
    sum_max_nums = max_train + max_val + max_test
    if max_num_per_class > sum_max_nums:
        diff = max_num_per_class - sum_max_nums
        max_train += diff
    
    return max_train, max_val, max_test


def get_train_val_test_indices(list_length):
    indices_ran = list(range(list_length))
    random.shuffle(indices_ran)
    train_len = int(list_length*.8)
    val_len = int(list_length*.1)
    test_len = int(list_length*.1)
    sum_indices = train_len + val_len + test_len
    if sum_indices != list_length:
        diff = list_length - sum_indices
        train_len += diff
    train_indices = []
    val_indices = []
    test_indices = []
    for i, item in enumerate(indices_ran):
        if i < train_len:
            train_indices.append(item)
        elif i >= train_len and i < train_len+val_len:
            val_indices.append(item)
        elif i >= train_len + val_len and i < list_length:
            test_indices.append(item)
    
    return train_indices, val_indices, test_indices


def make_dict_class_index(labels_class,labels_list):
    dict_class_index_list = {}
    for label in labels_class:
        dict_class_index_list[label] = []
        for i, label_item in enumerate(labels_list):
            if label == label_item:
                dict_class_index_list[label].append(i)
    
    return dict_class_index_list


def assign_indices_train_val_test(labels_class,dict_class_index,max_nums_train_val_test):
    dict_class_dataset_index_list = {}
    for label in labels_class:
        tot_indices = dict_class_index[label]
        tot_indices_copy = tot_indices.copy()
        random.shuffle(tot_indices_copy)
        train_indices = tot_indices_copy[:max_nums_train_val_test[0]]
        val_indices = tot_indices_copy[max_nums_train_val_test[0]:max_nums_train_val_test[0]+max_nums_train_val_test[1]]
        test_indices = tot_indices_copy[max_nums_train_val_test[0]+max_nums_train_val_test[1]:max_nums_train_val_test[0]+max_nums_train_val_test[1]+max_nums_train_val_test[2]]
        dict_class_dataset_index_list[label] = [train_indices,val_indices,test_indices]
    
    return dict_class_dataset_index_list


def check_4_dataset_mixing(labels_class,dict_class_dataset_index_list):
    train_indices = []
    test_indices = []
    val_indices = []
    for label in labels_class:
        label_indices = dict_class_dataset_index_list[label]
        train_indices.append(label_indices[0])
        val_indices.append(label_indices[1])
        test_indices.append(label_indices[2])
    try:
        for train_index in train_indices[0]:
            if train_index in test_indices or train_index in val_indices:
                raise DatasetMixing("Index {} of class {} is in multiple datasets".format(val_index, label))
        for val_index in val_indices[0]:
            if val_index in train_indices or val_index in test_indices:
                raise DatasetMixing("Index {} of class {} is in multiple datasets".format(val_index, label))
        for test_index in test_indices[0]:
            if test_index in train_indices or test_index in val_indices:
                raise DatasetMixing("Index {} of class {} is in multiple datasets".format(val_index, label))
        
        return True
    
    except DatasetMixing as e:
        print("DataPrepError: {}".format(e))
    
    return False
