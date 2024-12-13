# spotify-genre-classification

## Dataset
The dataset we used for this project was a Spotify dataset containing features such as track_id, artists, album_name, track_name, popularity, duration_ms, explicit, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature, and the target variable track_genre. This dataset presented a unique challenge with its 114 unique genres, requiring us to balance robust classification with computational efficiency. 

https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset

## Problem Overview
The goal of the project was to build a model to predict a track's genre using its audio features. This task posed several challenges, including the high dimensionality of the dataset and the presence of 114 distinct genres. 
