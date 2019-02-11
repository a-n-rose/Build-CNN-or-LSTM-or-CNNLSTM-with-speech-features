'''
These are a series of functions I made while working on a previous project (Mimic-Master).
It required noise cancellation and speech detection. It happens that these 
also help out in prepping speech for deep learning! 

I will work on adding explanations to each of the functions; I should have done that when I made them... hindsight is always 20-20.

'''

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
    
######

#noise reduction


def samps2stft(y, sr):
    if len(y)%2 != 0:
        y = y[:-1]
    #print("shape of samples: {}".format(y.shape))
    stft = librosa.stft(y)
    #print("shape of stft: {}".format(stft.shape))
    stft = np.transpose(stft)
    #print("transposed shape: {}".format(stft.shape))
    return stft


def stft2samps(stft,len_origsamp):
    #print("shape of stft: {}".format(stft.shape))
    istft = np.transpose(stft.copy())
    ##print("transposed shape: {}".format(istft.shape))
    samples = librosa.istft(istft,length=len_origsamp)
    return samples

def stft2power(stft_matrix):
    if stft_matrix is not None:
        if len(stft_matrix) > 0:
            stft = stft_matrix.copy()
            power = np.abs(stft)**2
            return power
        else:    
            raise TypeError("STFT Matrix is empty. Function 'stft2power' needs a non-empty matrix.")
    else:
        raise TypeError("STFT Matrix does not exist. Function 'stft2power' needs an existing matrix.")
    return None

    
def get_energy_rms(stft_matrix):
    #stft.shape[1] == bandwidths/frequencies
    #stft.shape[0] pertains to the time domain
    rms_list = [np.sqrt(sum(np.abs(stft_matrix[row])**2)/stft_matrix.shape[1]) for row in range(len(stft_matrix))]
    return rms_list

def get_mean_bandwidths(matrix_bandwidths):
    bw = matrix_bandwidths.copy()
    bw_mean = [np.mean(bw[:,bandwidth]) for bandwidth in range(bw.shape[1])]
    return bw_mean

def get_var_bandwidths(matrix_bandwidths):
    if len(matrix_bandwidths) > 0:
        bw = matrix_bandwidths.copy()
        bw_var = [np.var(bw[:,bandwidth]) for bandwidth in range(bw.shape[1])]
        return bw_var
    return None


def subtract_noise(noise_powerspec_mean,noise_powerspec_variance, speech_powerspec_row,speech_stft_row):
    npm = noise_powerspec_mean
    npv = noise_powerspec_variance
    spr = speech_powerspec_row
    stft_r = speech_stft_row.copy()
    for i in range(len(spr)):
        if spr[i] <= npm[i] + npv[i]:
            stft_r[i] = 1e-3
    return stft_r

def voice_activity_detection(stft, energy_matrix, energy_mean, start=True):
    voice_start,voice = sound_index(energy_matrix,energy_mean,start=True,)
    if voice:
        #print("Speech detected at index: {}".format(voice_start))
        stft = stft[voice_start:]
        
    else:
        print("No speech detected.")
    return stft

def rednoise(samples_recording,samples_noise, sampling_rate):
    '''
    calculates the power in noise signal and subtracts that
    from the recording
    
    returns recording samples with reduced noise
    '''
    
    #1) time domain to frequency domain:
    #get the short-time fourier transform (STFT) of noise and recording
    stft_n = samps2stft(samples_noise,sampling_rate)
    stft_r = samps2stft(samples_recording, sampling_rate)
    
    #2) calculate the power
    power_n = stft2power(stft_n)
    power_r = stft2power(stft_r)
    
    #3) calculate the power mean, and power variance of noise
    power_mean_n = get_mean_bandwidths(power_n)
    power_var_n = get_var_bandwidths(power_n)
    
    #4) subtract noise from recording:
    #using list comprehension to work through all samples of recording
    stft_r_rednoise = np.array([subtract_noise(power_mean_n,power_var_n,power_r[i],stft_r[i]) for i in range(stft_r.shape[0])])

    #5) detect speech and where when it starts:
    energy_r = get_energy_rms(stft_r)
    energy_r_mean = get_energy_mean(energy_r)
    stft_r_nr_vad = voice_activity_detection(stft_r_rednoise, energy_r, energy_r_mean)

    #save this to see if it worked:
    samps_rednoise_vad= stft2samps(stft_r_nr_vad, len(samples_recording))
    
    return samps_rednoise_vad
    
   
