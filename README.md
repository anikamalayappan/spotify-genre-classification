# spotify-genre-classification by JARS

## Dataset
The dataset we used for this project was a Spotify dataset containing features such as track_id, artists, album_name, track_name, popularity, duration_ms, explicit, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature, and the target variable track_genre. This dataset presented a unique challenge with its 114 unique genres, requiring us to balance robust classification with computational efficiency. 

https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset

## Problem Overview
The goal of the project was to build a model to predict a track's genre using its audio features. This task posed several challenges, including the high dimensionality of the dataset and the presence of 114 distinct genres. 

## Key Methodology
To address the problem, we began with extensive data preprocessing. We first dropped entries in the dataframe with the same track_id and track_name, but different genres to avoid confusing the model. Since the dataset was alphabetical based on genre, we first shuffled the rows and then kept the first entry for each duplicate song. We also dropped non-relevant identifiers such as track_id, artists, track_name, and album_name to focus only on audio-related features. Categorical variables like explicit were encoded into binary values (0/1), and numerical features were scaled using StandardScaler. To reduce class imbalance and improve model performance, we limited the dataset to the top 40 most frequent genres. This step significantly reduced the complexity of the classification task.

To address the problem, we decided to use random forests because as an ensemble method it is robust enough to be able to provide high accuracy without overfitting to the data, and it outperformed all other methods in accuracy score. The model's ability to handle non-linear relationships and automatically capture feature interactions also contributed to its good performance.

## Results
We performed a grid search using the GridSearchCV function provided by Scikit-learn for hyperparameter tuning of the random forest. The training data we passed in was from the train test split we obtained with the train_test_split function from Scikit-learn (80% training, 20% testing data). We used 5-fold cross validation as a part of the grid search, which returned the optimal random forest classifier with the parameters 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, and 'n_estimators': 150. Using this optimal classifier, we obtained y_predict values for the testing data, and found the accuracy score using the accuracy_score function from Scikit-learn. This yielded an accuracy of 63%.
Despite its performance, an accuracy of 63% is still not very high. It is evident that the Random Forest model still has limitations. First, it is computationally intensive, especially with a large dataset and many classes, which can make training and hyperparameter tuning time-consuming. Second, the model's predictions lack interpretability compared to simpler models like Logistic Regression, as it is difficult to discern the specific decision paths used for classification. Lastly, while Random Forest reduces overfitting compared to a single Decision Tree, it can still struggle with overfitting when the number of trees or depth is not properly tuned. These limitations highlight areas for further optimization or alternative approaches.


