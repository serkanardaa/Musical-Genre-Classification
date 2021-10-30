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

import utils

def indices_to_audio(indices_array):
#INPUTS:
#indices_array = array of indices in integer format
#OUTPUTS:
#audio_names_array = an array of audio file names for that specific indices array

    #convert to string
    indices_array_str = indices_array.astype(str)
    
    #adding missing zeros to the beginning
    same_length_indices = []
    for value in indices_array_str:
        orig_length = len(value)
        missing_zeros = 6 - orig_length
        new_value = missing_zeros*"0" + value
        same_length_indices.append(new_value)

    same_length_indices_array = np.array(same_length_indices)
    
    #adding .mp3 extension
    audio_names = []
    for name in same_length_indices_array:
        new_name = name + ".mp3"
        audio_names.append(new_name)

    audio_names_array = np.array(audio_names)
    
    return audio_names_array

def audio_to_indices(audio_names_array):
#INPUTS:
#audio_names_array = an array of audio file names for that specific indices array
#OUTPUTS:
#indices_array = array of indices in integer format

    indices_list = []

    for audio_name in audio_names_array:
        #removing .mp3 extension from audio name
        audio_name_extension = audio_name.split(".")
        audio_name = audio_name_extension[0]
        
        #removing 0s from the beginning of audio file
        index_string = audio_name.lstrip("0")
        
        #converting string index values to integer
        index_int = int(index_string)
        indices_list.append(index_int)
        
    #converting indices list to array
    indices_array = np.array(indices_list)
    
    return indices_array

#Existing genres are ['Hip-Hop', 'Pop', 'Rock', 'Folk', 'Experimental', 'Electronic', 'Classical', 'Old-Time / Historic' ]

def splitter(train_perc, valid_perc, test_perc, chosen_genres, tracks_dataframe):
#INPUTS
#train_perc = percentage of training data
#valid_perc = percentage of validation data
#test_perc = percentage of test data
#chosen_genres = array of chosen genres
#tracks_dataframe = loaded dataframe of tracks.csv metadata
    DATASET_DIR = "datasets/"
    for genre in chosen_genres:
        if genre == 'Old-Time / Historic':
            folder_name = "Old-Time - Historic"
        else:
            folder_name = genre
            # Getting folder of the genre to get track names
        path = os.path.join(DATASET_DIR, folder_name)
        path = path + "/"
        genre_tracks = np.array(os.listdir(path))

        if len(genre_tracks) != 0 and len(genre_tracks) != 3:

            #converting audio names to index values
            track_indices = audio_to_indices(genre_tracks)
            #size of the genre dataset in general
            dataset_size = len(track_indices)


            #size of training, validation, and test data according to given percentages
            training_size = int(np.floor(dataset_size * train_perc / 100))
            validation_size = int(np.floor(dataset_size * valid_perc / 100))
            test_size = int(np.floor(dataset_size * test_perc / 100))

            #getting tracks from the dataframe by using indices
            genreframe = tracks_dataframe.filter(items = track_indices, axis=0)

            #sorting columns according to artist name
            genreframe = genreframe.sort_values(by = ("artist", "name"))

            #getting file names for each dataset
            training_tracks = genreframe[:training_size]
            validation_tracks = genreframe[training_size: training_size + validation_size ]
            test_tracks = genreframe[training_size + validation_size : training_size + validation_size + test_size]

            if len(training_tracks) != 0:
                print("Distributing " + str(len(training_tracks)) + " tracks of " + genre + " genre to training folder...")
                #Getting indices of tracks from artist ordered genre dataframe
                training_indices = training_tracks.index
                #conversion of satisfying indices to array
                training_indices_array = training_indices.values
                #conversion of indices to audio files of the genre
                training_tracks = indices_to_audio(training_indices_array)

                #creating train folder if it does not exist
                training_folder = "train"
                train_path = path + training_folder
                is_train_path = os.path.isdir(train_path)

                if not is_train_path:     
                    os.mkdir(train_path)
                    print("Folder '% s' created for genre '% s'" % (training_folder, genre))
                else:
                    print("Folder '% s' already exists for genre '% s'" % (training_folder,genre))

                #Moving tracks from genre folder to genre/train folder
                for train_track in training_tracks:
                    #example: datasets/hip-hop/000002.mp3
                    train_track_source = path + train_track
                    #example: datasets/hip-hop/train/000002.mp3
                    train_track_target = train_path + "/" + train_track
                    os.replace(train_track_source, train_track_target)





            if len(validation_tracks) != 0:
                print("Distributing " + str(len(validation_tracks)) + " tracks of " + genre + " genre to validation folder...")
                #Getting indices of tracks from artist ordered genre dataframe
                validation_indices = validation_tracks.index
                #conversion of satisfying indices to array
                validation_indices_array = validation_indices.values
                #conversion of indices to audio files of the genre
                validation_tracks = indices_to_audio(validation_indices_array)

                #creating validation folder if it does not exist
                validation_folder = "validation"
                validation_path = path + validation_folder
                is_validation_path = os.path.isdir(validation_path)

                if not is_validation_path:     
                    os.mkdir(validation_path)
                    print("Folder '% s' created for genre '% s'" % (validation_folder, genre))
                else:
                    print("Folder '% s' already exists for genre '% s'" % (validation_folder,genre))

                #Moving tracks from genre folder to genre/validation folder
                for validation_track in validation_tracks:
                    validation_track_source = path + validation_track
                    validation_track_target = validation_path + "/" + validation_track
                    os.replace(validation_track_source, validation_track_target)


            if len(test_tracks) != 0:
                print("Distributing " + str(len(test_tracks)) + " tracks of " + genre + " genre to test folder...")
                #Getting indices of tracks from artist ordered genre dataframe
                test_indices = test_tracks.index
                #conversion of satisfying indices to array
                test_indices_array = test_indices.values
                #conversion of indices to audio files of the genre
                test_tracks = indices_to_audio(test_indices_array)

                #creating test folder if it does not exist
                test_folder = "test"
                test_path = path + test_folder
                is_test_path = os.path.isdir(test_path)

                if not is_test_path:     
                    os.mkdir(test_path)
                    print("Folder '% s' created for genre '% s'" % (test_folder, genre))
                else:
                    print("Folder '% s' already exists for genre '% s'" % (test_folder,genre))

                #Moving tracks from genre folder to genre/test folder
                for test_track in test_tracks:
                    test_track_source = path + test_track
                    test_track_target = test_path + "/" + test_track
                    os.replace(test_track_source, test_track_target)                    

def data_reset(chosen_genres):
#INPUT
#chosen_genres = array of chosen genres for resetting
    DATASET_DIR = "datasets/"
    for genre in chosen_genres:
        if genre == 'Old-Time / Historic':
            folder_name = "Old-Time - Historic"
        else:
            folder_name = genre

        # Getting folder of the genre
        path = os.path.join(DATASET_DIR, folder_name)
        # Specifying train, validation, and test folder paths 
        train_path = path + "/train"
        validation_path = path + "/validation"
        test_path = path + "/test"

        is_train_path = os.path.isdir(train_path)
        is_validation_path = os.path.isdir(validation_path)
        is_test_path = os.path.isdir(test_path)

        if is_train_path:
            train_tracks = np.array(os.listdir(train_path))
            if len(train_tracks)!= 0:
                print("Resetting training data of " + genre + " genre")
                for train_track in train_tracks:

                    #example: datasets/hip-hop/train/000002.mp3
                    train_track_source = train_path + "/" + train_track
                    #example: datasets/hip-hop/000002.mp3
                    train_track_target = path + "/" + train_track
                    os.replace(train_track_source, train_track_target)


        if is_validation_path:
            validation_tracks = np.array(os.listdir(validation_path))
            if len(validation_tracks)!= 0:
                print("Resetting validation data of " + genre + " genre")
                for validation_track in validation_tracks:
                    validation_track_source = validation_path + "/" + validation_track
                    validation_track_target = path + "/" + validation_track
                    os.replace(validation_track_source, validation_track_target)



        if is_test_path:
            test_tracks = np.array(os.listdir(test_path))
            if len(test_tracks)!= 0:
                print("Resetting test data of " + genre + " genre")
                for test_track in test_tracks:
                    test_track_source = test_path + "/" + test_track
                    test_track_target = path + "/" + test_track
                    os.replace(test_track_source, test_track_target)


        #removes training validation and test data folders
        os.rmdir(train_path)

        os.rmdir(validation_path)

        os.rmdir(test_path)


def show_statistics(audio_file_array, tracks_dataframe):
#INPUTS: 
#audio_file_array = a numpy array that contains audio file names
#tracks_dataframe = dataframe to get the statistical information about audio files. For example tracks.csv file dataframe

    audio_name_indices = audio_to_indices(audio_file_array)
    indices_to_df = tracks_dataframe.filter(items = audio_name_indices, axis=0)
    print("+++ The data contains: " + str(len(audio_name_indices)) + " tracks")
    
    
    genre = np.array(indices_to_df["track"]["genre_top"].unique())
    print("+++ The genre is: ", genre)
    
    artists = np.array(indices_to_df["artist"]["name"].unique())
    print("+++ The data contains "+ str(len(artists)) +" artists ")
    
    albums = np.array(indices_to_df["album"]["title"].unique())
    print("+++ The data contains " + str(len(albums)) + " albums" )
    
def common_artists(audio_file_array1, audio_file_array2, tracks_dataframe):
#INPUTS: 
#audio_file_array1 = a numpy array that contains audio file names 
#audio_file_array2= a numpy array that contains audio file names to compare common artists
#tracks_dataframe = dataframe to get the statistical information about audio files. For example tracks.csv file dataframe
    audio_name_indices1 = audio_to_indices(audio_file_array1)
    indices_to_df1 = tracks_dataframe.filter(items = audio_name_indices1, axis=0)
    artists1 = set(np.array(indices_to_df1["artist"]["name"].unique()) )
    
    audio_name_indices2 = audio_to_indices(audio_file_array2)
    indices_to_df2 = tracks_dataframe.filter(items = audio_name_indices2, axis=0)
    artists2 = set(np.array(indices_to_df2["artist"]["name"].unique())  )
    
    common_artists = artists1.intersection(artists2)
    print("+++ The first dataset contains "+ str(len(artists1)) +" artists ")
    print("+++ The second dataset contains "+ str(len(artists2)) +" artists ")
    print("+++ The datasets have " + str(len(common_artists)) + " common artists")