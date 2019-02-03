import sys
import os
import numpy as np
import time
import datetime

import feature_extraction_scripts.organize_speech_data as orgdata
import feature_extraction_scripts.feature_extraction_functions as featfun
from feature_extraction_scripts.errors import FeatureExtractionFail

#to keep saved files unique
#include their names with a timestamp
def get_date():
    time = datetime.datetime.now()
    time_str = "{}y{}m{}d{}h{}m{}s".format(time.year,time.month,time.day,time.hour,time.minute,time.second)
    return(time_str)

def main(data_path,feature_type,num_filters=None,delta=False,noise=False,vad=False,timesteps=None,context_window=None,noise_path=None,limit=None):
    #set defaults:
    if num_filters is None:
        num_filters = 40
    if timesteps is None:
        timesteps = 5
    if context_window is None:
        context_window = 5
    if delta:
        num_features = num_filters * 3
    else:
        num_features = num_filters
    #frame_width is the sum of frames including:
    #1 central frame with a context window in front and one context window behind
    frame_width = context_window*2 + 1

    #####################################################################
    ######################## HOUSE KEEPING ##############################
    
    start = time.time()
    time_stamp = get_date()
    head_folder_beg = "./ml_speech_projects/"
    curr_folder = "features_and_models_{}".format(time_stamp)
    head_folder = head_folder_beg+curr_folder
    #create folder to store all data (encoded labels, features)
    if not os.path.exists(head_folder):
        os.makedirs(head_folder)
        
    '''
    Collect all labels in data:
    Labels should be the subdirectories of the data directory
    Not included:
    Folders/files with names:
    * starting with "_"
    * are typical GitHub files, like LICENSE
    '''
    labels_class = orgdata.collect_labels(data_path)

    '''
    Create labels-encoding dictionary:
    This helps when saving data later to npy files
    Integer encode the labels and save with feature data as label column
    '''
    dict_labels_encoded = orgdata.create_save_dict_labels_encode(labels_class,head_folder)
    

    train_val_test_filenames = []
    train_val_test_directories = []
    for i in ["train","val","test"]:
        new_path = "{}/data_{}/".format(head_folder,i)
        train_val_test_filenames.append(new_path+"{}_features".format(i))
        train_val_test_directories.append(new_path)
        try:
            os.makedirs(new_path)
        except OSError as e:
            print("Directory  ~  {}  ~  already exists".format(new_path))
            pass

    
    #############################################
    ############## DATA ORGANIZATION ############

    #collect filenames and labels of each filename
    paths, labels_wavefile = orgdata.collect_audio_and_labels(data_path)

    #to balance out the classes, find the label/class w fewest recordings
    max_num_per_class, class_max_samps = orgdata.get_max_samples_per_class(labels_class,labels_wavefile)
    
    #LOG THE SETTINGS OF FEATURE EXTRACTION IN CSV FILE
    dict_info_feature_extraction = {"data path":data_path,"limit":limit,"features":feature_type,"num original features":num_filters,"num total features":num_features,"delta":delta,"noise":noise,"beginning silence removal":vad,"timesteps":timesteps,"context window":context_window,"num classes":len(labels_class),"time stamp":time_stamp}
    orgdata.log_extraction_settings(dict_info_feature_extraction,head_folder)
    orgdata.log_class4balance(max_num_per_class,class_max_samps,head_folder)
    '''
    Create dictionary with labels and their indices in the lists: labels_wavefile and paths
    useful in separating the indices into balanced train, validation, and test datasets
    '''
    dict_class_index_list = orgdata.make_dict_class_index(labels_class,labels_wavefile)
    
    '''
    Assign number of recordings for each dataset, 
    keeping the data balanced between classes
    Defaults:
    * .8 of max number of samples --> train
    * .1 of max number of samples --> validation
    * .1 of max number of samples --> test
    '''
    max_nums_train_val_test = orgdata.get_max_nums_train_val_test(max_num_per_class)

    #randomly assign indices (evenly across class) to train, val, test datasets:
    dict_class_dataset_index_list = orgdata.assign_indices_train_val_test(labels_class,dict_class_index_list,max_nums_train_val_test)

    #make sure no indices mix between datasets:
    orgdata.check_4_dataset_mixing(labels_class,dict_class_dataset_index_list)


    #############################################
    ############# FEATURE EXTRACTION ############


    start_feature_extraction = time.time()

    try:
        
        for i in range(3):
            dataset_index = i   # 0 = train, 1 = validation, 2 = test
            
            extraction_completed = featfun.save_feats2npy(labels_class,dict_labels_encoded,train_val_test_filenames[dataset_index],max_nums_train_val_test[dataset_index],dict_class_dataset_index_list,paths,labels_wavefile,feature_type,num_filters,num_features,timesteps,frame_width,head_folder,limit=limit,delta=delta,noise_wavefile=noise_path,vad=vad,dataset_index=dataset_index)
            
            if extraction_completed:
                print("\nRound {} feature extraction successful.\n".format(i+1))
            else:
                print("\nRound {} feature extraction was unsuccessful.".format(i+1))
                raise FeatureExtractionFail()
            
        end_feature_extraction = time.time()
        print("Duration of feature extraction: {} minutes".format(round((end_feature_extraction-start_feature_extraction)/60,2)))
    
        print("\nTo train a model, copy and paste the following into the model training script:".upper())
        print("\n\n'{}'\n\n".format(curr_folder))
        
        return True
    
    except FeatureExtractionFail:
        print("Feature Extraction Error. Terminated feature extraction process.")
        pass
    
    return False
    
if __name__=="__main__":

    #variables to set:
    
    #which directory has the data?
    data_path = "./data"
    #should there be a limit on how many waves are processed?
    limit = .05 #False or fraction of data to be extracted
    #which type of features to extract?
    feature_type = "fbank" # "mfcc" TO DEBUG: "stft"
    #number of filters or coefficients?
    num_filters = 40 # 13, None
    delta = False # Calculate the 1st and 2nd derivatives of features?
    noise = True # Add noise to speech data?
    vad = True #voice activity detection
    timesteps = 5
    context_window = 5
    noise_path = "./data/_background_noise_/doing_the_dishes.wav" # None

    main(
        data_path,feature_type,
        num_filters=num_filters,delta=delta,noise=noise,vad=vad,
        timesteps=timesteps,context_window=context_window,noise_path=noise_path,limit = limit
        )
