# spotify-genre-classification

## Dataset
The dataset we used for this project was a Spotify dataset containing features such as track_id, artists, album_name, track_name, popularity, duration_ms, explicit, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature, and the target variable track_genre. This dataset presented a unique challenge with its 114 unique genres, requiring us to balance robust classification with computational efficiency. 

https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset

## Problem Overview
The goal of the project was to build a model to predict a track's genre using its audio features. This task posed several challenges, including the high dimensionality of the dataset and the presence of 114 distinct genres. 

## Key Methodology
To address the problem, we began with extensive data preprocessing. We first dropped entries in the dataframe with the same track_id and track_name, but different genres to avoid confusing the model. Since the dataset was alphabetical based on genre, we first shuffled the rows and then kept the first entry for each duplicate song. We also dropped non-relevant identifiers such as track_id, artists, track_name, and album_name to focus only on audio-related features. Categorical variables like explicit were encoded into binary values (0/1), and numerical features were scaled using StandardScaler. To reduce class imbalance and improve model performance, we limited the dataset to the top 40 most frequent genres. This step significantly reduced the complexity of the classification task.

To address the problem, we decided to use random forests because as an ensemble method it is robust enough to be able to provide high accuracy without overfitting to the data, and it outperformed all other methods in accuracy score.

