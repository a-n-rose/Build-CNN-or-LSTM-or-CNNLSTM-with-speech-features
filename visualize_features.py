"""
This script allows the user to visually explore the parameters for
feature extraction.
Enter in any wavefile you would like to explore.
"""


import os
from feature_extraction_scripts.feature_extraction_functions import save2png


if __name__ == "__main__":

    # VARIABLES FOR THE USER TO SET:

    # "cat"
    # wav = "./data/cat/0e5193e6_nohash_1.wav"

    # "bird"
    # wav = "./data/bird/0a7c2a8d_nohash_1.wav"

    # "house"
    # wav = "./data/happy/0c5027de_nohash_0.wav"

    # "seven"
    wav = "./data/seven/0cd323ec_nohash_0.wav"

    # variables to set and see how they change the picture
    # which type of features to extract?
    # "mfcc" "fbank" "stft"
    feature_type = "fbank"

    # number of filters or coefficients? If STFT, doesn't matter..
    # can put None
    # Options: FBANK: 40, 20 - MFCC: 40, 20, 13 - STFT: None
    num_filters = 40
    if feature_type == "stft":
        num_filters = 201

    # Calculate the 1st and 2nd derivatives of features?
    delta = False

    # Kinda sorta... Pitch (dominant frequency)
    dom_freq = False

    # Add noise to speech data?
    noise = True

    # If noise == True, put the pathway to that noise here:
    if noise:
        noise_path = "./data/_background_noise_/doing_the_dishes.wav"
    else:
        noise_path = None
    # NOTE: as it is, noise is applied at a scale from 0 to 0.75
    # you can change the scale or keep it constant to stay the same if
    # you'd like.
    # Change this in: function "apply_noise" which is located in the
    # script "feature_extraction_functions.py"

    # Apply voice activity detection (removes the beginning and ending
    # 'silence'/background noise of recordings)
    vad = True
    timesteps = 5
    context_window = 5
    frame_width = context_window * 2 + 1

    # SETTINGS THE SCRIPT ASSIGNS ITSELF
    if delta:
        num_feature_columns = num_filters * 3
    else:
        num_feature_columns = num_filters

    if dom_freq:
        num_feature_columns += 1

    # create folder to store all data (encoded labels, features)
    visuals_folder = "visualizations"
    if not os.path.exists(visuals_folder):
        os.makedirs(visuals_folder)

    save2png(
        timesteps,
        frame_width,
        wav,
        feature_type,
        num_filters,
        num_feature_columns,
        visuals_folder,
        delta=delta,
        dom_freq=dom_freq,
        noise_wavefile=noise_path,
        vad=vad,
    )
