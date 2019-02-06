'''
These are a series of functions to transform the speech form from the time and amplitude domain (waveform) to the frequency and power domain (MFCC, FBAN, STFT).

Note: these functions would be better set up within a Class.
For this workshop I thought leaving them as individual functions would be more straightforward but I don't think that is the case :P 
'''

#save info
import csv
import sys
from pathlib import Path

#audio 
import librosa
import librosa.display
import matplotlib.pyplot as plt

#data prep
import numpy as np
import random

#my own speech prep: voice activity detection
import feature_extraction_scripts.prep_noise as prep_data_vad_noise
from feature_extraction_scripts.errors import NoSpeechDetected, LimitTooSmall,FeatureExtractionFail



 
#delta
def get_change_acceleration_rate(spectro_data):
    #first derivative = delta (rate of change)
    delta = librosa.feature.delta(spectro_data)
    #second derivative = delta delta (acceleration changes)
    delta_delta = librosa.feature.delta(spectro_data,order=2)

    return delta, delta_delta

    
#load wavefile, set settings for that
def get_samps(wavefile,sr=None,high_quality=None):
    if sr is None:
        sr = 16000
    if high_quality:
        quality = "kaiser_high"
    else:
        quality = "kaiser_fast"
    y, sr = librosa.load(wavefile,sr=sr,res_type=quality) 
    
    return y, sr

#set settings for mfcc extraction
def get_mfcc(y,sr,num_mfcc=None,window_size=None, window_shift=None):
    '''
    set values: default for MFCCs extraction:
    - 40 MFCCs
    - windows of 25ms 
    - window shifts of 10ms
    '''
    if num_mfcc is None:
        num_mfcc = 40
    if window_size is None:
        n_fft = int(0.025*sr)
    else:
        n_fft = int(window_size*0.001*sr)
    if window_shift is None:
        hop_length = int(0.010*sr)
    else:
        hop_length = int(window_shift*0.001*sr)
    mfccs = librosa.feature.mfcc(y,sr,n_mfcc=num_mfcc,hop_length=hop_length,n_fft=n_fft)
    mfccs = np.transpose(mfccs)
    
    return mfccs

#get fbank, and set settings for that
def get_mel_spectrogram(y,sr,num_mels = None,window_size=None, window_shift=None):
    '''
    set values: default for mel spectrogram calculation (FBANK)
    - windows of 25ms 
    - window shifts of 10ms
    '''
    if num_mels is None:
        num_mels = 40
    if window_size is None:
        n_fft = int(0.025*sr)
    else:
        n_fft = int(window_size*0.001*sr)
    if window_shift is None:
        hop_length = int(0.010*sr)
    else:
        hop_length = int(window_shift*0.001*sr)
        
    fbank = librosa.feature.melspectrogram(y,sr,n_fft=n_fft,hop_length=hop_length,n_mels=num_mels)
    fbank = np.transpose(fbank)
    
    return fbank

#get stft and adjust settings if you'd like 
#note: I have not messed around with the window_size or shift here
#if you change these, you might have to adjust the default number of feature 
#columns assigned to stft in the main module (see right below def main())
def get_stft(y,sr,window_size=None, window_shift=None):
    if window_size is None:
        n_fft = int(0.025*sr)
    else:
        n_fft = int(window_size*0.001*sr)
    if window_shift is None:
        hop_length = int(0.010*sr)
    else:
        hop_length = int(window_shift*0.001*sr)
    stft = np.abs(librosa.stft(y,n_fft=n_fft,hop_length=hop_length)) #comes in complex numbers.. have to take absolute value
    stft = np.transpose(stft)
    
    return stft

#super experimental. I wanted fundamental frequency but this was easier
def get_domfreq(y,sr):
    '''
    collecting the frequencies with highest magnitude
    '''
    frequencies, magnitudes = get_freq_mag(y,sr)
    #select only frequencies with largest magnitude, i.e. dominant frequency
    dom_freq_index = [np.argmax(item) for item in magnitudes]
    dom_freq = [frequencies[i][item] for i,item in enumerate(dom_freq_index)]
    
    return np.array(dom_freq)

#get a collection of frequencies at the same windows as other extraction techniques i.e. 25ms with 10ms shifts (which is standard for much research)
#this can be adjusted here.. this script is prepared for these window settings
#it might work with others but I haven't tested that yet.
def get_freq_mag(y,sr,window_size=None, window_shift=None):
    '''
    default values:
    - windows of 25ms 
    - window shifts of 10ms
    '''
    if window_size is None:
        n_fft = int(0.025*sr)
    else:
        n_fft = int(window_size*0.001*sr)
    if window_shift is None:
        hop_length = int(0.010*sr)
    else:
        hop_length = int(window_shift*0.001*sr)
    #collect frequencies present and their magnitudes
    frequencies,magnitudes = librosa.piptrack(y,sr,hop_length=hop_length,n_fft=n_fft)
    frequencies = np.transpose(frequencies)
    magnitudes = np.transpose(magnitudes)
    
    return frequencies, magnitudes

#saving a lot of features in the exact shape I wanted was easiest done with .npy files. It's fast to save and fast to load.
def save_feats2npy(labels_class,dict_labels_encoded,data_filename4saving,max_num_samples,dict_class_dataset_index_list,paths_list,labels_list,feature_type,num_filters,num_feature_columns,time_step,frame_width,head_folder,limit=None,delta=False,dom_freq=False,noise_wavefile=None,vad=False,dataset_index=None):
    if dataset_index is None:
        dataset_index = 0
    #dataset_index represents train (0), val (1) or test (2) datasets

    #create empty array to fill with values
    if limit:
        max_num_samples = int(max_num_samples*limit)
        expected_rows = max_num_samples*len(labels_class)*frame_width*time_step
    else:
        expected_rows = max_num_samples*len(labels_class)*frame_width*time_step
    feats_matrix = np.zeros((expected_rows,num_feature_columns+1)) # +1 for the label
    
    #update the user what's going on:
    msg = "\nFeature Extraction: Section {} of 3\nNow extracting features: {} wavefiles per class.\nWith {} classes, processing {} wavefiles.\nFeatures will be saved in the file {}.npy\n\n".format(dataset_index+1,max_num_samples,len(labels_class),len(labels_class)*max_num_samples,data_filename4saving)
    print(msg)
    
    #go through all data in dataset and fill in the matrix
    row = 0
    #this row indicates how far along the empty matrix is getting filled
    completed = False
    #if the functions ends early, it will return that it was not completed.
    
    try:
        if expected_rows < 1*frame_width*time_step:
            #I once set the limit to 0 by accident... which doesn't make sense.
            raise LimitTooSmall("\nIncrease Limit: The limit at '{}' is too small.".upper().format(limit))
        
        #put the paths of the waves and their labels together in a list of tuples --> make sure they don't get separated!
        #this list will then get iterated through, and each wavefile/label pair will be processed together
        paths_labels_list_dataset = []
        for i, label in enumerate(labels_class):
            '''
            Note: I balanced the data based on class/label. Therefore,
            wavefiles in each class have been assigned, equally and at 
            random, to train, validaton, and test datasets. 
            I know this is a bit confusing but....
            here I am collecting all of the wavefile and label pairs 
            belonging to each class, for each section (train, val, test).
            This function collects those pairs only for one of those sections 
            at a time: the 'dataset_index' variable here represents the 
            current section (i.e. 0 == train, 1 == validation, 2 == test)
            '''
            train_val_test_index_list = dict_class_dataset_index_list[label]
            
            for k in train_val_test_index_list[dataset_index]:
                paths_labels_list_dataset.append((paths_list[k],labels_list[k]))
        
        #shuffle indices:
        #this is important!! Otherwise the algorithm will learn based on 
        #label/class order (as I ordered the list above by class)
        random.shuffle(paths_labels_list_dataset)
        
        for wav_label in paths_labels_list_dataset:

            if row >= feats_matrix.shape[0]:
                # This means we've filled the matrix! Yaay!
                break
            else:
                wav_curr = wav_label[0]
                label_curr = wav_label[1]
                #integer encode the label:
                label_encoded = dict_labels_encoded[label_curr]
                
                #function below basically extracts the features and makes sure each sample's features are the same size: they are cut short
                #if too long and zero padded if too short
                feats = coll_feats_manage_timestep(time_step,frame_width,wav_curr,feature_type,num_filters,num_feature_columns,head_folder,delta=delta,dom_freq=dom_freq, noise_wavefile=noise_wavefile,vad = vad)
                
                #add label column - need label to stay with the features!
                label_col = np.full((feats.shape[0],1),label_encoded)
                feats = np.concatenate((feats,label_col),axis=1)
                
                #fill the matrix with the features just collected
                feats_matrix[row:row+feats.shape[0]] = feats
                
                #actualize the row for the next set of features to fill it with
                row += feats.shape[0]
                
                #print on screen the progress
                progress = row / expected_rows * 100
                sys.stdout.write("\r%d%% through current section" % progress)
                sys.stdout.flush()
        print("\nRow reached: {}\nSize of matrix: {}\n".format(row,feats_matrix.shape))
        completed = True
    
    except LimitTooSmall as e:
        print(e)

    finally:
        np.save(data_filename4saving+".npy",feats_matrix)
        
    return completed


#this function feeds variables on to the feature extraction function 'get_feats' (and it shapes the data to the same size). 
#It is also a beautiful example of why classes are great. I chose not to do a class for these functions because I thought it would be more straightforward, for a workshop setting....
def coll_feats_manage_timestep(time_step,frame_width,wav,feature_type,num_filters,num_feature_columns,head_folder,delta=False,dom_freq=False,noise_wavefile=None,vad = True):
    feats = get_feats(wav,feature_type,num_filters,num_feature_columns,head_folder,delta=delta,dom_freq=dom_freq,noise_wavefile=noise_wavefile,vad = vad)
    max_len = frame_width*time_step
    if feats.shape[0] < max_len:
        diff = max_len - feats.shape[0]
        feats = np.concatenate((feats,np.zeros((diff,feats.shape[1]))),axis=0)
    else:
        feats = feats[:max_len,:]
    
    return feats

#noise
#at default applies it at varying strengths. You can set it to a certain level here:
def apply_noise(y,sr,wavefile):
    #at random apply varying amounts of environment noise
    rand_scale = random.choice([0.0,0.25,0.5,0.75])
    #rand_scale = 0.75
    if rand_scale > 0.0:
        total_length = len(y)/sr
        y_noise,sr = librosa.load(wavefile,sr=16000)
        envnoise_normalized = prep_data_vad_noise.normalize(y_noise)
        envnoise_scaled = prep_data_vad_noise.scale_noise(envnoise_normalized,rand_scale)
        envnoise_matched = prep_data_vad_noise.match_length(envnoise_scaled,sr,total_length)
        if len(envnoise_matched) != len(y):
            diff = int(len(y) - len(envnoise_matched))
            if diff < 0:
                envnoise_matched = envnoise_matched[:diff]
            else:
                envnoise_matched = np.append(envnoise_matched,np.zeros(diff,))
        y += envnoise_matched

    return y

#collects the actual features, according to the settings assigned
#such as with noise, voice activity detection/beginning silence removal, etc.
#mfcc, fbank, stft, delta, dom_freq
def get_feats(wavefile,feature_type,num_features,num_feature_columns,head_folder,delta=False,dom_freq=False,noise_wavefile = None,vad = False):
    y, sr = get_samps(wavefile)

    if vad:
        try:
            y, speech = prep_data_vad_noise.get_speech_samples(y,sr)
            if speech:
                pass
            else:
                raise NoSpeechDetected("\n!!! FYI: No speech was detected in file: {} !!!\n".format(wavefile))
        except NoSpeechDetected as e:
            print("\n{}".format(e))
            filename = '{}/no_speech_detected.csv'.format(head_folder)
            with open(filename,'a') as f:
                w = csv.writer(f)
                w.writerow([wavefile])
            
    if noise_wavefile:
        y = apply_noise(y,sr,noise_wavefile)
        
    extracted = []
    if "mfcc" in feature_type.lower():
        extracted.append("mfcc")
        features = get_mfcc(y,sr,num_mfcc=num_features)
        #features -= (np.mean(features, axis=0) + 1e-8)
        if delta:
            delta, delta_delta = get_change_acceleration_rate(features)
            features = np.concatenate((features,delta,delta_delta),axis=1)
    elif "fbank" in feature_type.lower():
        extracted.append("fbank")
        features = get_mel_spectrogram(y,sr,num_mels = num_features)
        #features -= (np.mean(features, axis=0) + 1e-8)
        if delta:
            delta, delta_delta = get_change_acceleration_rate(features)
            features = np.concatenate((features,delta,delta_delta),axis=1)
    elif "stft" in feature_type.lower():
        extracted.append("stft")
        features = get_stft(y,sr)
        #features -= (np.mean(features, axis=0) + 1e-8)
        if delta:
            delta, delta_delta = get_change_acceleration_rate(features)
            features = np.concatenate((features,delta,delta_delta),axis=1)
    if dom_freq:
        dom_freq = get_domfreq(y,sr)
        dom_freq = dom_freq.reshape((dom_freq.shape+(1,)))
        features = np.concatenate((features,dom_freq),axis=1)
    if features.shape[1] != num_feature_columns: 
        raise FeatureExtractionFail("The file '{}' results in the incorrect  number of columns (should be {} columns): shape {}".format(wavefile,num_feature_columns,features.shape))
    
    return features


#only for visualization purproses: save to png how the features look. Used in 'visualize_features.py' script
def save2png(time_step,frame_width,wav,feature_type,num_filters,num_feature_columns,head_folder,delta=False,dom_freq=False,noise_wavefile=None,vad = True):
    feats = coll_feats_manage_timestep(time_step,frame_width,wav,feature_type,num_filters,num_feature_columns,head_folder,delta=delta,dom_freq=dom_freq,noise_wavefile=noise_wavefile,vad = vad)
    
    #transpose the features to go from left to right in time:
    feats = np.transpose(feats)
    
    #create graph and save to png
    plt.clf()
    librosa.display.specshow(feats)
    if noise_wavefile:
        noise = True
    else:
        noise = False
    plt.title("{}: {} timesteps, frame width of {}".format(wav,time_step,frame_width))
    plt.tight_layout(pad=0)
    pic_path = "{}{}_vad{}_noise{}_delta{}_domfreq{}".format(feature_type,num_feature_columns,vad,noise,delta,dom_freq)
    path = unique_path(Path(head_folder), pic_path+"{:03d}.png")
    plt.savefig(path)

    return True

def unique_path(directory, name_pattern):
    counter = 0
    while True:
        counter += 1
        path = directory / name_pattern.format(counter)
        if not path.exists():
            return path

