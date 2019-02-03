#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 18:28:19 2018

@author: airos
"""

import numpy as np
import librosa
import math
import random

from feature_extraction_scripts.errors import NoSpeechDetected

    
def match_length(noise,sr,desired_length):
    noise2 = np.array([])
    final_noiselength = sr*desired_length
    original_noiselength = len(noise)
    frac, int_len = math.modf(final_noiselength/original_noiselength)
    for i in range(int(int_len)):
        noise2 = np.append(noise2,noise)
    if frac:
        max_index = int(original_noiselength*frac)
        end_index = len(noise) - max_index
        rand_start = random.randrange(0,end_index)
        noise2 = np.append(noise2,noise[rand_start:rand_start+max_index])
    if len(noise2) != final_noiselength:
        diff = int(final_noiselength - len(noise2))
        if diff < 0:
            noise2 = noise2[:diff]
        else:
            noise2 = np.append(noise2,np.zeros(diff,))
    return(noise2)

def normalize(array):
    max_abs = max(abs(array))
    if max_abs > 1:
        mult_var = 1.0/max_abs
        array_norm = array*mult_var
        return(array_norm)
    else:
        return(array)

def scale_noise(np_array,factor):
    '''
    If you want to reduce the amplitude by half, the factor should equal 0.5
    '''
    return(np_array*factor)

def wave2stft(np_array,sr):
    stft = librosa.stft(np_array,hop_length=int(0.01*sr),n_fft=int(0.025*sr))
    stft = np.transpose(stft)
    return stft
    
def get_energy(stft):
    rms_list = [np.sqrt(sum(np.abs(stft[row])**2)/stft.shape[1]) for row in range(len(stft))]
    return rms_list

def get_energy_mean(rms_energy):
    energy_mean = sum(rms_energy)/len(rms_energy)
    return energy_mean

  
def suspended_energy(speech_energy,speech_energy_mean,row,start):
    try:
        if start == True:
            if row <= len(speech_energy)-4:
                if speech_energy[row+1] and speech_energy[row+2] and speech_energy[row+3] > speech_energy_mean:
                    return True
        else:
            if row >= 3:
                if speech_energy[row-1] and speech_energy[row-2] and speech_energy[row-3] > speech_energy_mean:
                    return True
    except IndexError as ie:
        return False

def sound_index(speech_energy,speech_energy_mean,start = True):
    if start == True:
        side = 1
        beg = 0
        end = len(speech_energy)
    else:
        side = -1
        beg = len(speech_energy)-1
        end = -1
    for row in range(beg,end,side):
        if speech_energy[row] > speech_energy_mean:
            if suspended_energy(speech_energy, speech_energy_mean, row,start=start):
                if start==True:
                    #to catch plosive sounds
                    while row >= 0:
                        row -= 1
                        row -= 1
                        if row < 0:
                            row = 0
                        break
                    return row, True
                else:
                    #to catch quiet consonant endings
                    while row <= len(speech_energy):
                        row += 1
                        row += 1
                        if row > len(speech_energy):
                            row = len(speech_energy)
                        break
                    return row, True
    else:
        #print("No Speech Detected")
        pass
    return beg, False

    
def get_speech_samples(samples, sr):
    try:
        signal_length = len(samples)
        stft = wave2stft(samples,sr)
        energy = get_energy(stft)
        energy_mean = get_energy_mean(energy)
        beg = sound_index(energy,energy_mean,start=True)
        end = sound_index(energy,energy_mean,start=False)
        if beg[1] == False or end[1] == False:
            raise NoSpeechDetected("No speech detected")
        perc_start = beg[0]/len(energy)
        perc_end = end[0]/len(energy)
        sample_start = int(perc_start*signal_length)
        sample_end = int(perc_end*signal_length)
        samples_speech = samples[sample_start:sample_end]
        
        return samples_speech, True
    
    except NoSpeechDetected as e:
        pass
        
    return samples, False
    
        
    
    
   
