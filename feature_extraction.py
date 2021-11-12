import os
import shutil
import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import librosa
import librosa.display
import pydub as dub
import pickle
from varname import nameof
import itertools as it
import utils
from scipy import stats

# Creating audio segment from audio file

def audio_loader(audio_file): # Function for loading audio samples and sampling frequency
# Gets input audio_file location and outputs attributes of that audio file
    audio_name, format_type = audio_file.split(".")
    audio = dub.AudioSegment.from_file(audio_file, format=format_type)
    channel_count = audio.channels
    # Choosing one of the channels
    channels = audio.split_to_mono()
    audio = channels[0]
    
    fs = audio.frame_rate                  # sampling frequency
    num_frame = audio.frame_count()        # number of samples
    time = np.arange(num_frame) / fs        # timevector of length num_frame
    total_dur = num_frame / fs              # total duration of audio
    samples = audio.get_array_of_samples() # getting samples from audio
    max_amp = audio.max_possible_amplitude # maximum amplitude in samples
    data = np.divide(samples, max_amp)       # normalizing

    return data, fs

def windowing(a_win_ms, h_len_ms, fs): # function for windowing audios
#Gets input analysis window size and hop_size in terms of ms and calculates sample lengths along with overlap in samples

    a_len = int(np.ceil( fs * a_win_ms /1000 )) # ms to number of samples
    h_len = int(np.ceil( fs * h_len_ms /1000 ))
    overlap = a_len - h_len # in samples

    return a_len, h_len, overlap

def genre_feature_extractor(chosen_genres,tracks_list, dataset_type, a_win_ms = 46, h_len_ms = 23, window_type = "hann"): # function for extracting features
    
#INPUTS
#chosen_genres = an list of genre names
#tracks_list = a list of tracks from different genres. 
#dataset_type = type of data (train, validation, test)
#a_win_ms = size of analysis window in miliseconds. Defeault = 46 ms
#h_len_ms = size of hop length in miliseconds. Defatult = 23 ms
#window_type = window type for the frames. Default = hann

#OUTPUTS
#genre_feature_list = a list that each element contains features calculated from all frames in all songs for a genre
#genre_song_length_list = a list that each element is a list that contains frame length (seconds) 
#of each song for each genre to use
#in majority voting in future

#frame_labels = a list that each element is an array of labels for each frame in each song. Each element is from another genre
#song_labels = an array that contains genre label of each song from each genre. 

    genre_feature_list = []
    genre_song_length_list = []
    frame_labels = []
    song_labels = []

    #Printing labels of each genre
    labels = list(range(len(chosen_genres)))
    for l in range(len(labels)):
        print("Label of " + chosen_genres[l] + " is " + str(l))
    
    #Looping over genres
    DATASET_DIR = "datasets/"
    for label, genre in enumerate(chosen_genres):
        song_length_list = []
        feature_list = []
        if genre == 'Old-Time / Historic':
            folder_name = "Old-Time - Historic"
        else:
            folder_name = genre
            
        # Getting folder of the genre
        path = os.path.join(DATASET_DIR, folder_name)
        folder_path = path + "/" + dataset_type
        
        #Checking if genre has tracks 
        if len(tracks_list[label]) != 0:
            #looping over each track
            for num, track in enumerate(tracks_list[label]):

                if (num + 1) % 50 == 0:
                    print("Features of " + str(num + 1) + " songs are extracted in " + genre + " genre")
                #Calculating stft of each track
                track_path = folder_path + "/" + track
                data, fs = audio_loader(track_path)
                a_len, h_len, overlap = windowing(a_win_ms, h_len_ms, fs)
                spec = np.abs(librosa.stft(data, n_fft=a_len, hop_length=h_len, win_length=a_len, window=window_type))
                
                #Feature extractions
                
                f_SCent = librosa.feature.spectral_centroid(sr = fs, S=spec, n_fft=a_len,hop_length=h_len )

                f_SRoll = librosa.feature.spectral_rolloff(sr = fs,S=spec, n_fft=a_len,hop_length=h_len)

                f_SFlat = librosa.feature.spectral_flatness(S=spec, n_fft=a_len,hop_length=h_len)

                f_SBand = librosa.feature.spectral_bandwidth(sr = fs,S=spec, n_fft=a_len,hop_length=h_len)

                f_SCont = librosa.feature.spectral_contrast(sr = fs,S=spec, n_fft=a_len,hop_length=h_len)

                f_ZC = librosa.feature.zero_crossing_rate(data, frame_length=a_len, hop_length=h_len)

                f_MFCC = librosa.feature.mfcc(data,fs, n_fft=a_len, hop_length=h_len ,n_mfcc = 5)
                
                # Averaging features of frames over 1 second
                
                frames_per_second = int(np.ceil( (1000 - a_win_ms + h_len_ms) / h_len_ms) )
                #Concatenated features
                f_conc = np.concatenate((f_SCent,f_SRoll,f_SFlat,f_SBand,f_SCont,f_ZC,f_MFCC))
                #Number of features
                num_features = f_conc.shape[0]
                #Number of frames
                num_frames = f_conc.shape[1]
                #Number of seconds
                num_secs = num_frames / frames_per_second
                #Boolean if there are residual frames in the end that is not exactly 1 second
                is_residual = num_secs > int(num_secs)
                #Number of frames excluding residual frames in the end
                num_frames_no_res = frames_per_second * int(num_secs)
                
                f_conc_reshape = f_conc[:,:num_frames_no_res].reshape(num_features * int(num_secs),frames_per_second)
                f_conc_reshape_mean = np.mean(f_conc_reshape, axis = 1)
                f_conc_reshape_std = np.std(f_conc_reshape, axis = 1)
                f_conc_mean = f_conc_reshape_mean.reshape(num_features, int(num_secs))
                f_conc_std = f_conc_reshape_std.reshape(num_features, int(num_secs))
                
                #if there are residuals frames remaining in the end, calculate their avg and st and combine with prev.
                if is_residual:
                    f_conc_res = f_conc[:,num_frames_no_res:]
                    f_conc_res_mean = np.mean(f_conc_res, axis = 1).reshape(num_features, 1)
                    f_conc_res_std = np.std(f_conc_res, axis = 1).reshape(num_features, 1)
                    f_conc_mean = np.concatenate((f_conc_mean, f_conc_res_mean), axis = 1)
                    f_conc_std = np.concatenate((f_conc_std, f_conc_res_std), axis = 1)
                
                #feature array with 17 * 2 feature rows (doubles because of mean, var) and num of secs + residual sec columns
                song_features = np.concatenate((f_conc_mean,f_conc_std))
                
                feature_list.append(song_features)
                song_length_list.append(song_features.shape[1])
                #adding label of the song to song_labels list
                song_labels.append(label)
                
            #adding feature set of the genre
            genre_feature_list.append(np.hstack(feature_list))
            #adding list of lengths of the songs in seconds for the genre
            genre_song_length_list.append(song_length_list)
            #adding labels of seconds in all of the current genre with the same column length of feature matrix
            frame_labels.append(np.ones(np.hstack(feature_list).shape[1]) * label )
        else:
            print("Genre "+ genre + " has no tracks to extract features.")
            
    return  genre_feature_list, genre_song_length_list, frame_labels, song_labels

def feature_label_con(list_of_arrays, is_feature = True):
#INPUT
#list_of_arrays = a list that contains feature matrices or labels elements for each genre
#is_feature = a boolean indicates whether the list of arrays is features or labels

#OUTPUT
#conc_array = concatenated features or labels
    conc_array = list_of_arrays[0]

    if is_feature:
        for i in range(1, len(list_of_arrays)):
            conc_array = np.concatenate((conc_array,list_of_arrays[i]), axis = 1)
    else:
        for i in range(1, len(list_of_arrays)):
            conc_array = np.concatenate((conc_array,list_of_arrays[i]))

    return conc_array

def feature_saver(genre_feature_list,name_gfl, genre_song_length_list, name_gsll, frame_labels, name_fl, song_labels, name_sl):
# INPUTS
# genre_feature_list = genre_feature_list file
# name_gfl = name for genre_feature_list file (string)
# genre_song_length_list = genre_song_length_list file
# name_gsll = name for genre_song_length_list file (string)
# frame_labels = frame_labels file
# name_fl = name for frame_labels file (string)
# song_labels = song_labels file
# name_sl = name for song_labels file (string)

    # Saving feature files of input data
    with open("features/" + name_gfl + ".npy", "wb") as fp:
        pickle.dump(genre_feature_list, fp)
    with open("features/" + name_gsll + ".npy", "wb") as fp:
        pickle.dump(genre_song_length_list, fp)
    with open("features/" + name_fl + ".npy", "wb") as fp:
        pickle.dump(frame_labels, fp)
    with open("features/" + name_sl + ".npy", "wb") as fp:
        pickle.dump(song_labels, fp)
        
def feature_loader(genre_feature_list, genre_song_length_list, frame_labels, song_labels):
# INPUTS
# genre_feature_list = name of genre_feature_list file
# genre_song_length_list = name of genre_song_length_list file
# frame_labels = name of frame_labels file
# song_labels = name of song_labels file

    # Loading feature files of input data
    with open("features/" + genre_feature_list + ".npy", 'rb') as fp:
        genre_feature_list_load = pickle.load(fp)
    with open("features/" +genre_song_length_list + ".npy", 'rb') as fp:
        genre_song_length_list_load = pickle.load(fp)
    with open("features/" +frame_labels + ".npy", 'rb') as fp:
        frame_labels_load = pickle.load(fp)
    with open("features/" +song_labels + ".npy", 'rb') as fp:
        song_labels_load = pickle.load(fp)
        
    return genre_feature_list_load, genre_song_length_list_load, frame_labels_load, song_labels_load

def majority_vote(y_pred_frame, y_slen_l):
#INPUT
#y_pred_frame = prediction labels for each frame outputed from classifier (numpy array)
#y_slen_l = length of each song to calculate majority voting from corresponding frames (list of lists from each genre)

#OUTPUT
#y_pred_song = array containing majority voted labels for each song from their corresponding frames
    #concetaneting genres to one list
    conc_y_slen_l =list(it.chain.from_iterable(y_slen_l))
    print(len(conc_y_slen_l))
    #list that will contain song labels for predicted labels of corresponding frames
    y_pred_song = []
    start = 0
    end = 0
    count = 0
    for slen in conc_y_slen_l:
        count += 1
        end += slen
        #finding the most repeating label in frames of current song
        m = stats.mode( y_pred_frame[start:end] )
        if len(m[0]) != 0:
            label = int(m[0][0])
            #Adding label to song labels list
            y_pred_song.append(label)
            #increasing start point for next song
        else:# in rare cases ( only once ) we just add one additional 0 to increment 399 labels to 400
            y_pred_song.append(0)
        
        start += slen
    
    y_pred_song = np.array(y_pred_song)
    
    return y_pred_song