# Musical-Genre-Classification
Musical Genre Classification by using bayesian and deep learning classifiers
This repository contains the project for musical genre recognition by using Librosa features such as spectral features and MFCC in order to distinguish different genres.
For the dataset, we have used FMA dataset: https://github.com/mdeff/fma which provides a huge number of tracks from several genres.

The contents of the files in this repository are as in the following:

main.ipynb: Implementation of the project from processing data to results of classification of genres with several classifiers from sklearn such as SVC, Gaussian Naive Bayes,
KNN, and MLP.

data_processing_guide.ipynb: Guide for how data_processing.py functions work.

data_processing.py: Contains functions for processing the data such as partitioning, removal of corrupted data, showing statistical information of tracks and conversion between dataframe
and audio files.

feature_extraction.py: Contains functions for extracting features from tracks such as loading audio, windowing for stft, feature extraction and feature loading, feature saving,
and majority voting for deciding on label of songs.

