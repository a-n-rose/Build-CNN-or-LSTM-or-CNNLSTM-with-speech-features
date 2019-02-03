#save info
import csv
import sys

#audio 
import librosa
import librosa.display
import matplotlib.pyplot as plt

#data prep
import numpy as np
import random

#my own speech prep: voice activity detection
import feature_extraction_scripts.prep_noise as prep_data_vad_noise
from feature_extraction_scripts.errors import NoSpeechDetected, LimitTooSmall

 
 
def get_change_acceleration_rate(spectro_data):
    #first derivative = delta (rate of change)
    delta = librosa.feature.delta(spectro_data)
    #second derivative = delta delta (acceleration changes)
    delta_delta = librosa.feature.delta(spectro_data,order=2)

    return delta, delta_delta


def apply_noise(y,sr,wavefile):
    #at random apply varying amounts of environment noise
    rand_scale = random.choice([0.0,0.25,0.5,0.75])
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


def get_feats(wavefile,feature_type,num_features,head_folder,delta=False,noise_wavefile = None,vad = False):
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
    if delta:
        num_feature_columns = num_features*3
    else:
        num_feature_columns = num_features
        
    extracted = []
    if "mfcc" in feature_type.lower():
        extracted.append("mfcc")
        features = get_mfcc(y,sr,num_mfcc=num_features)
        features -= (np.mean(features, axis=0) + 1e-8)
        if delta:
            delta, delta_delta = get_change_acceleration_rate(features)
            features = np.concatenate((features,delta,delta_delta),axis=1)
    elif "fbank" in feature_type.lower():
        extracted.append("fbank")
        features = get_mel_spectrogram(y,sr,num_mels = num_features)
        features -= (np.mean(features, axis=0) + 1e-8)
        if delta:
            delta, delta_delta = get_change_acceleration_rate(features)
            features = np.concatenate((features,delta,delta_delta),axis=1)
    ###!!!!!!! Need to Debug..
    #elif "stft" in feature_type.lower():
        #extracted.append("stft")
        #features = get_stft(y,sr)
        #features -= (np.mean(features, axis=0) + 1e-8)
        #if delta:
            #delta, delta_delta = get_change_acceleration_rate(features)
            #features = np.concatenate((features,delta,delta_delta),axis=1)
    if features.shape[1] != num_feature_columns: 
        raise FeatureExtractionError("The file '{}' results in the incorrect  number of columns (should be {} columns): shape {}".format(wavefile,num_features,features.shape))
    
    return features
    

def get_samps(wavefile,sr=None,high_quality=None):
    if sr is None:
        sr = 16000
    if high_quality:
        quality = "kaiser_high"
    else:
        quality = "kaiser_fast"
    y, sr = librosa.load(wavefile,sr=sr,res_type=quality) 
    
    return y, sr


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


def get_stft(y,sr,window_size=None, window_shift=None):
    if window_size is None:
        n_fft = int(0.025*sr)
    else:
        n_fft = int(window_size*0.001*sr)
    if window_shift is None:
        hop_length = int(0.010*sr)
    else:
        hop_length = int(window_shift*0.001*sr)
    stft = np.abs(librosa.feature.stft(y,sr,n_fft=n_fft,hop_length=hop_length)) #comes in complex numbers.. have to take absolute value
    stft = np.transpose(stft)
    
    return stft


def get_domfreq(y,sr):
    frequencies, magnitudes = get_freq_mag(y,sr)
    #select only frequencies with largest magnitude, i.e. dominant frequency
    dom_freq_index = [np.argmax(item) for item in magnitudes]
    dom_freq = [frequencies[i][item] for i,item in enumerate(dom_freq_index)]
    
    return dom_freq


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


def save_feats2npy(labels_class,dict_labels_encoded,data_filename4saving,max_num_samples,dict_class_dataset_index_list,paths_list,labels_list,feature_type,num_filters,num_features,time_step,frame_width,head_folder,limit=None,delta=False,noise_wavefile=None,vad=False,dataset_index=None):
    if dataset_index is None:
        dataset_index = 0

    #create empty array to fill with values
    if limit:
        max_num_samples = int(max_num_samples*limit)
        expected_rows = max_num_samples*len(labels_class)*frame_width*time_step
    else:
        expected_rows = max_num_samples*len(labels_class)*frame_width*time_step
    
    feats_matrix = np.zeros((expected_rows,num_features+1)) # +1 for the label
    #go through all data in dataset and fill in the matrix
    
    
    msg = "\nFeature Extraction: Section {} of 3\nNow extracting features: {} wavefiles per class.\nWith {} classes, processing {} wavefiles.\nFeatures will be saved in the file {}.npy\n\n".format(dataset_index+1,max_num_samples,len(labels_class),len(labels_class)*max_num_samples,data_filename4saving)
    print(msg)
    
    row = 0
    completed = False
    
    try:
        if expected_rows < 1*frame_width*time_step:
            raise LimitTooSmall("\nIncrease Limit: The limit at '{}' is too small.".upper().format(limit))
        paths_labels_list_dataset = []
        for i, label in enumerate(labels_class):
            #labels_list_dataset = []
            train_val_test_index_list = dict_class_dataset_index_list[label]
            #print(train_val_test_index_list[dataset_index])
            for k in train_val_test_index_list[dataset_index]:
                paths_labels_list_dataset.append((paths_list[k],labels_list[k]))
        
        #shuffle indices:
        random.shuffle(paths_labels_list_dataset)
        
        for wav_label in paths_labels_list_dataset:

            if row >= feats_matrix.shape[0]:
                break
            else:
                wav_curr = wav_label[0]
                label_curr = wav_label[1]
                label_encoded = dict_labels_encoded[label_curr]
                feats = coll_feats_manage_timestep(time_step,frame_width,wav_curr,feature_type,num_filters,head_folder,delta=delta, noise_wavefile=noise_wavefile,vad = vad)
                #add label column:
                label_col = np.full((feats.shape[0],1),label_encoded)
                feats = np.concatenate((feats,label_col),axis=1)
                
                feats_matrix[row:row+feats.shape[0]] = feats
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
    
    
def coll_feats_manage_timestep(time_step,frame_width,wav,feature_type,num_filters,head_folder,delta=False,noise_wavefile=None,vad = True):
    feats = get_feats(wav,feature_type,num_filters,head_folder,delta=delta,noise_wavefile=noise_wavefile,vad = vad)
    max_len = frame_width*time_step
    if feats.shape[0] < max_len:
        diff = max_len - feats.shape[0]
        feats = np.concatenate((feats,np.zeros((diff,feats.shape[1]))),axis=0)
    else:
        feats = feats[:max_len,:]
    
    return feats
        
