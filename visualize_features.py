'''
This script allows the user to visually explore the parameters for feature extraction. Enter in any wavefile you would like to explore.
'''


import os
from feature_extraction_scripts.feature_extraction_functions import save2png



if __name__=="__main__":
    
    
    #VARIABLES FOR THE USER TO SET:
    
    #an example wave: Sheila
    wav = "./data/sheila/1c1060b1_nohash_0.wav"
    #variables to set and see how they change the picture
    #which type of features to extract?
    feature_type = "stft" # "mfcc" "fbank" "stft"
    #number of filters or coefficients? If STFT, doesn't matter.. can put None
    num_filters = 40 # Options: FBANK: 40, 20 - MFCC: 40, 20, 13 - STFT: None
    if feature_type == "stft":
        num_filters = 201
    delta = False # Calculate the 1st and 2nd derivatives of features?
    dom_freq = False # Kinda sorta... Pitch (dominant frequency)
    noise = True # Add noise to speech data?
    #If noise == True, put the pathway to that noise here:
    if noise:
        noise_path = "./data/_background_noise_/doing_the_dishes.wav" 
    else:
        noise_path = None
    vad = True # Apply voice activity detection (removes the beginning and ending 'silence'/background noise of recordings)
    timesteps = 5
    context_window = 5
    
    
    #SETTINGS THE SCRIPT ASSIGNS ITSELF
    if delta:
        num_feature_columns = num_filters*3
    else:
        num_feature_columns = num_filters
    if dom_freq:
        num_feature_columns += 1
    frame_width = context_window*2+1

    #create folder to store all data (encoded labels, features)
    visuals_folder = "visualizations"
    if not os.path.exists(visuals_folder):
        os.makedirs(visuals_folder)
    
    
    save2png(timesteps,frame_width,wav,feature_type,num_filters,num_feature_columns,visuals_folder,delta=delta,dom_freq=dom_freq,noise_wavefile=noise_path,vad = vad)
