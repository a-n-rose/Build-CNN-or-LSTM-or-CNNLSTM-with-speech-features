# To control the system (exit the program, create new directories)
import sys
import os

# Keep track of the time --> time stamps; keep track of duration
import time
import datetime

# python library to handle matrices
import numpy as np

# import modules with functions I wrote to organize data
# and extract features
import feature_extraction_scripts.organize_speech_data as orgdata
import feature_extraction_scripts.feature_extraction_functions as featfun

# Handle exceptions I think might happen
from feature_extraction_scripts.errors import FeatureExtractionFail, ExitApp


##########################################################################
########################### FUNCTIONS ####################################


def get_date():
    """
    This creates a string of the day, hour, minute and second
    I use this to make folder names unique

    For the files themselves, I generate genuinely unique names:
    i.e. name001.csv, name002.csv, etc.
    """
    time = datetime.datetime.now()
    time_str = (f"{time.year}y{time.month}m{time.day}d"
                f"{time.hour}h{time.minute}m{time.second}s")
    return time_str


def main(data_path, feature_type, num_filters=None, delta=False,
         dom_freq=False, noise=False, vad=False, timesteps=None,
         context_window=None, noise_path=None, limit=None):
    """
    Here defaults are set, if they aren't yet set by the user.

    Number of Filters OR Number of Coefficients

    FBANK or mel filterbank energies are produced based on the number
    of mel filters used
        20 and 40 are pretty common

    MFCCs or mel frequency cepstral coefficients also use mel filters.
    With additional mathematic equations applied, to ultimately make
    speech easier for traditional machine learning algorithms to learn
    (they reduce the colinearity of the features)
        13,20,40 coefficients are not uncommon.
    """
    if not num_filters and feature_type != "stft":
        # Common number of coefficients / filters for
        # MFCC and FBANK is 40
        num_filters = 40
    if feature_type == "stft":
        # This number might change if you change the
        # window size/ window shift settings within the feature
        # extraction functions.
        # I have put window size == 25ms and
        # window shift == 10ms as default.
        num_filters = 201
    if not noise:
        # ensure that if the user doesn't want to add noise to data,
        # that the wavefile variable is set to None
        noise_path = None
    if not timesteps:
        # For the LSTM: in how many segments do you want each recording
        # to be split up? The LSTM would be fed each section,
        # consecutively.
        timesteps = 5
    if not context_window:
        # This concerns how large you want each set of frames to be.
        # The frame width contains a central frame, with surrounding
        # context windows.
        context_window = 5

    # keep track of how many feature columns will be needed to store
    # feature data
    if delta:
        # delta stands for both delta and delta delta these are the
        # first and second derivatives of whatever features are
        # extracted apparently these show 'rate of change' and
        # 'rate of acceleration' and have been used in speech
        # recognition/related machine learning tasks
        num_feature_columns = num_filters * 3
    else:
        num_feature_columns = num_filters
    if dom_freq is not False:
        # dominant frequency is an experimental parameter...
        # simply which frequency is the dominant one, at each of the
        # sampled windows and window shifts.
        num_feature_columns += 1

    frame_width = context_window * 2 + 1

    #####################################################################
    ######################## HOUSE KEEPING ##############################

    start = time.time()
    time_stamp = get_date()
    head_folder_beg = "./ml_speech_projects/"
    curr_folder = f"{feature_type}{num_feature_columns}_models_{time_stamp}"
    head_folder = head_folder_beg + curr_folder
    # create folder to store all data (encoded labels, features)
    if not os.path.exists(head_folder):
        os.makedirs(head_folder)

    """
    Collect all labels in data:
    Labels should be the subdirectories of the data directory
    Not included:
    Folders/files with names:
    * starting with "_"
    * are typical GitHub files, like LICENSE
    """
    try:
        print(f"Features being collected: {feature_type.upper()}")
        print(f"Total number of feature columns: {num_feature_columns}")
        labels_class = orgdata.collect_labels(data_path)
        labels_print = ""
        for i in range(len(labels_class)):
            x = "\n{}) {}".format(i + 1, labels_class[i])
            labels_print += x
        print(f"Classes found: {labels_print}")
        print(f"Limit: {limit}")
        print("Is this correct? (Y/n)")
        answer = False
        while not answer:
            correct_label = input().lower()
            if correct_label == "y" or correct_label == "":
                answer = True
            elif correct_label == "n":
                raise ExitApp()
            else:
                print('Please enter "y" or "n"')

        """
        Create labels-encoding dictionary:
        This helps when saving data later to npy files
        Integer encode the labels and save with feature data as label column
        """
        dict_labels_encoded = orgdata.create_save_dict_labels_encode(
            labels_class, head_folder
        )

        # create directories to store train, validation, and test data
        train_val_test_filenames = []
        train_val_test_directories = []
        for i in ["train", "val", "test"]:
            new_path = f"{head_folder}/data_{i}/"
            train_val_test_filenames.append(os.path.join(new_path,
                                                         f"{i}_features"))
            train_val_test_directories.append(new_path)
            try:
                os.makedirs(new_path)
            except OSError as e:
                print(f"Directory  ~  {new_path}  ~  already exists")
                pass

        #############################################
        ############## DATA ORGANIZATION ############

        """
        I made the decision to balance the classes out.
        Real data doesn't have balanced classes.
        But, because this code was prepared for a workshop, one for
        exploring how feature extraction influenced deep learning
        models, I decided to remove that confounding factor.
        """

        # collect filenames and labels of each filename
        paths, labels_wavefile = orgdata.collect_audio_and_labels(data_path)

        # to balance out the classes, find the label/class with
        # fewest recordings
        max_num_per_class, class_max_samps = orgdata.get_max_samples_per_class(
            labels_class, labels_wavefile
        )

        """
        Create dictionary with labels and their indices in the lists:
        labels_wavefile and paths useful in separating the indices into
        balanced train, validation, and test datasets.
        """
        dict_class_index_list = orgdata.make_dict_class_index(
            labels_class, labels_wavefile
        )

        """
        Assign number of recordings for each dataset,
        keeping the data balanced between classes
        Defaults:
        * .8 of max number of samples --> train
        * .1 of max number of samples --> validation
        * .1 of max number of samples --> test
        """
        max_nums_train_val_test = orgdata.get_max_nums_train_val_test(max_num_per_class)

        # randomly assign indices (evenly across class) to train,
        # val, test datasets:
        dict_class_dataset_index_list = orgdata.assign_indices_train_val_test(
            labels_class, dict_class_index_list, max_nums_train_val_test
        )

        # make sure no indices mix between datasets:
        orgdata.check_4_dataset_mixing(labels_class, dict_class_dataset_index_list)
    except ExitApp:
        sys.exit(0)
    except IndexError:
        print("Error collecting data.\n".upper())
        print("Double check how your data is organized."
              "Are the labels printed above correct?")
        sys.exit(-1)

    #############################################
    ############# FEATURE EXTRACTION ############

    start_feature_extraction = time.time()

    try:
        for i in range(3):
            dataset_index = i  # 0 = train, 1 = validation, 2 = test

            # see the script 'feature_extraction_functions.py' in the
            # folder 'feature_extraction_scripts' for more information
            # about extraction.
            extraction_completed = featfun.save_feats2npy(
                labels_class,
                dict_labels_encoded,
                train_val_test_filenames[dataset_index],
                max_nums_train_val_test[dataset_index],
                dict_class_dataset_index_list,
                paths,
                labels_wavefile,
                feature_type,
                num_filters,
                num_feature_columns,
                timesteps,
                frame_width,
                head_folder,
                limit=limit,
                delta=delta,
                dom_freq=dom_freq,
                noise_wavefile=noise_path,
                vad=vad,
                dataset_index=dataset_index,
            )

            if extraction_completed:
                print(f"Round {i+1} feature extraction successful.")
            else:
                print(f"Round {i+1} feature extraction was unsuccessful.")
                raise FeatureExtractionFail()

        end_feature_extraction = time.time()
        extraction_duration = round((end_feature_extraction - start_feature_extraction) / 60, 2)
        print(f"Duration of feature extraction: {extraction_duration} minutes")

        # LOG THE SETTINGS OF FEATURE EXTRACTION IN CSV FILE
        dict_info_feature_extraction = {
            "data path": data_path,
            "limit": limit,
            "features": feature_type,
            "num original features": num_filters,
            "num total features": num_feature_columns,
            "delta": delta,
            "dominant frequency": dom_freq,
            "noise": noise,
            "beginning silence removal": vad,
            "timesteps": timesteps,
            "context window": context_window,
            "num classes": len(labels_class),
            "time stamp": time_stamp,
            "duration in minutes": extraction_duration,
        }
        orgdata.log_extraction_settings(dict_info_feature_extraction, head_folder)
        orgdata.log_class4balance(max_num_per_class, class_max_samps, head_folder)

        print("Folder name to copy and paste for the training script:".upper())
        print("\n\n'{}'\n\n".format(curr_folder))

        return True

    except FeatureExtractionFail as e:
        print(e)
        print("Feature Extraction Error. Terminated feature extraction process.")
        pass

    return False


if __name__ == "__main__":

    # variables to set:

    # which directory has the data?
    data_path = "./data"

    # should there be a limit on how many waves are processed?
    # Options: False or fraction of data to be extracted
    limit = 0.05

    # which type of features to extract?
    # "mfcc" "fbank" "stft"
    feature_type = "fbank"

    # number of filters or coefficients? If STFT, doesn't matter..
    # can put None
    # Options: 40, 20, 13, None
    num_filters = 40

    # Calculate the 1st and 2nd derivatives of features?
    delta = False

    # Kinda sorta... Pitch (dominant frequency)
    dom_freq = False

    # Add noise to speech data?
    noise = True

    # Apply voice activity detection (removes the beginning and ending
    # 'silence'/background noise of recordings)
    vad = True

    timesteps = 5
    context_window = 5

    # If noise == True, put the pathway to that noise here:
    noise_path = "./data/_background_noise_/doing_the_dishes.wav"

    main(
        data_path,
        feature_type,
        num_filters=num_filters,
        delta=delta,
        noise=noise,
        vad=vad,
        dom_freq=dom_freq,
        timesteps=timesteps,
        context_window=context_window,
        noise_path=noise_path,
        limit=limit,
    )
